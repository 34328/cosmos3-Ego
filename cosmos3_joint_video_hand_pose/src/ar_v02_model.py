"""Joint chunk-conditioned training on native Cosmos packed samples."""

import math
import weakref
import torch
from .ar_model import ARStepContext, EgoVerseARModel, OmniMoTCausalModel
from .ar_v02_attention import JointTeacherForcingAttention
from .ar_v02_layout import STATE, CONDITION_VIDEO, LAYOUT_VERSION, JointChunkLayout
from .ar_v02_packing import pack_joint_sequence


class EgoVerseARV02Model(EgoVerseARModel):
    _required_tf_frames_per_chunk = 4

    def _is_chunkwise_tf(self):
        # Cosmos' native [V0,V1..VC] grid differs from our repeated U/S layout.
        # Keep native truncation/assertion off: JointChunkLayout owns C=4 and
        # preserves the last partial chunk, including every future action row.
        return False

    def __init__(self, config, chunk_state_conditioning=True, seed=42, action_representation="legacy_local_delta_absolute_hand_v1"):
        if action_representation not in ("legacy_local_delta_absolute_hand_v1", "fixed_camera_wrist_local_delta_latent_v1"):
            raise ValueError("unsupported action representation")
        self.action_representation = action_representation
        if (
            not config.diffusion_expert_config.enable_action_state_embedding
            or not config.diffusion_expert_config.enable_vision_condition_embedding
        ):
            raise ValueError("v0.2 requires state and condition-image type embeddings")
        if config.teacher_forcing_detach_clean_kv or not config.diffusion_expert_config.enable_fps_modulation:
            raise ValueError("v0.2 requires clean KV gradients and FPS modulation")
        if not chunk_state_conditioning:
            raise ValueError("state ablation is deferred; complete v0.2 includes every S_k")
        super().__init__(
            config, train_chunk_sizes=(4,), train_window_range=(15, 15), seed=seed, whole_action_loss=True
        )
        self.chunk_state_conditioning = True
        self._joint_layout = None
        self._joint_layouts = []

    def _sample_step_context(self, iteration: int) -> ARStepContext:
        """V0.2 training uses a fixed four-latent chunk and 15-chunk history."""
        return ARStepContext(chunk_size=4, window=15)

    def _encode_vision_x0_tokens(
        self,
        raw_state_vision,
        num_vision_items_per_sample,
        vision_condition_indexes,
        num_views_per_vision_item=None,
        balance_vae_encode=False,
    ):
        if self._ar_step is None:
            raise ValueError("choose C before VAE encoding")
        if num_vision_items_per_sample is not None or num_views_per_vision_item is not None:
            raise ValueError("joint chunks expect one monocular RGB stream per sample")
        self._joint_original_frames = []
        result = []
        # Each encode resets causal VAE context; never encode the complete clip first.
        for raw in raw_state_vision:
            t = raw.shape[2]
            if (t - 1) % 4:
                raise ValueError("RGB clip must contain 1+4N sampled frames")
            original = 1 + (t - 1) // 4
            self._joint_original_frames.append(original)
            layout = JointChunkLayout(original, 1, self._ar_step.chunk_size)
            chunks = []
            for b in layout.boundaries:
                start = b.source_start // 2
                stop = b.source_stop // 2 + 1
                with torch.no_grad():
                    z = self._encode_vision_item(raw[:, :, start:stop], num_views=1)
                expected = 1 + b.latent_stop - b.latent_start
                if z.shape[2] != expected:
                    raise ValueError("causal VAE returned unexpected chunk length")
                chunks.append(z)
            result.append(torch.cat(chunks, dim=2))
        return result

    def _prepare_training_data(self, data_batch, iteration):
        if self._ar_step is None or self._ar_step.window != 15:
            raise ValueError("choose C and H=15 before preparation")
        versions = data_batch.get("ar_layout_version")
        while isinstance(versions, (list, tuple)):
            if not versions:
                raise ValueError("missing layout version")
            if all(isinstance(x, str) for x in versions):
                if any(x != LAYOUT_VERSION for x in versions):
                    raise ValueError("incompatible AR layout")
                versions = versions[0]
                break
            versions = versions[0]
        if versions != LAYOUT_VERSION or "ar_boundary_states" not in data_batch:
            raise ValueError("joint_chunk_cond_v1 dataset required; legacy state layout is not interchangeable")
        representation = getattr(self, "action_representation", "legacy_local_delta_absolute_hand_v1")
        observed = data_batch.get("ar_action_representation")
        def flatten(value):
            if isinstance(value, (list, tuple)):
                return [item for child in value for item in flatten(child)]
            return [value]
        if representation == "fixed_camera_wrist_local_delta_latent_v1":
            if self._ar_step.chunk_size != 4 or any(x != representation for x in flatten(observed)):
                raise ValueError("fixed-camera actions require C=4 and explicit matching representation metadata")
        elif observed is not None and any(x != representation for x in flatten(observed)):
            raise ValueError("new action representation cannot be passed to a legacy model")
        result = super()._prepare_training_data(data_batch, iteration)
        _, plans, data, memory_info, _, _ = result
        if memory_info["skip_text"]:
            raise ValueError("training needs complete sample texts")
        from .model import _visibility_from_batch

        vis = _visibility_from_batch(data_batch)
        states = data_batch["ar_boundary_states"]
        if isinstance(states, torch.Tensor):
            states = list(states)
        if len(states) != data.batch_size or len(vis) != data.batch_size:
            raise ValueError("one state and visibility payload per sample required")
        layouts = []
        out_vis = []
        for i, (video, future) in enumerate(zip(data.x0_tokens_vision, data.x0_tokens_action, strict=True)):
            _, _, t, h, w = video.shape
            p = self.config.diffusion_expert_config.patch_spatial
            layout = JointChunkLayout(
                self._joint_original_frames[i], math.ceil(h / p) * math.ceil(w / p), self._ar_step.chunk_size
            )
            if t != layout.num_video_frames:
                raise ValueError("VAE/layout mismatch")
            st = states[i]
            while isinstance(st, list):
                if len(st) != 1:
                    raise ValueError("invalid nested state payload")
                st = st[0]
            while st.ndim > 2 and st.shape[0] == 1:
                st = st.squeeze(0)
            data.x0_tokens_action[i], mask = layout.assemble_action(future.reshape(-1, 64), st, vis[i])
            layouts.append(layout)
            out_vis.append(mask)
        data.raw_state_action = None
        if data.action_valid_mask is not None and any(x is not None for x in data.action_valid_mask):
            raise ValueError("external action_valid_mask conflicts with the 57D contract")
        self._current_hand_visibility = out_vis
        self._joint_layouts = layouts
        self._joint_layout = layouts[0] if len(layouts) == 1 else None
        return result

    def _pack_input_sequence(
        self,
        sequence_plans,
        input_text_indexes,
        gen_data_clean,
        input_timesteps,
        include_end_of_generation_token=False,
        skip_text_tokens=False,
        initial_mrope_temporal_offset=0,
    ):
        if skip_text_tokens or include_end_of_generation_token:
            raise ValueError("invalid training text packing")
        layouts = self._joint_layouts
        if len(layouts) == 1 and self._joint_layout is not None:
            layouts = [self._joint_layout]
        cfg = self.config.diffusion_expert_config
        return pack_joint_sequence(
            layout=layouts,
            gen_data_clean=gen_data_clean,
            text_ids=input_text_indexes,
            special_tokens=self.llm_special_tokens,
            timesteps=input_timesteps,
            latent_patch_size=cfg.patch_spatial,
            condition_frames=[() for _ in layouts],
            base_fps=cfg.base_fps,
            reset_spatial=cfg.unified_3d_mrope_reset_spatial_ids,
            modality_margin=cfg.unified_3d_mrope_temporal_modality_margin,
            initial_temporal_offset=initial_mrope_temporal_offset,
        )

    def _get_train_noise_level_vision(
        self, batch_size, is_image_batch, num_vision_latent_frames, resolutions=None, num_tokens=None, iteration=None
    ):
        layouts = self._joint_layouts
        n = max(len(x.boundaries) for x in layouts) + 1

        def repeat(x):
            if x is None or isinstance(x, str):
                return x
            return [v for v in x for _ in range(n)]

        ts, sg = OmniMoTCausalModel._get_train_noise_level_vision(
            self,
            batch_size=batch_size * n,
            is_image_batch=False,
            num_vision_latent_frames=repeat(num_vision_latent_frames),
            resolutions=repeat(resolutions),
            num_tokens=repeat(num_tokens),
            iteration=iteration,
        )
        ts, sg = ts.reshape(batch_size, n), sg.reshape(batch_size, n)
        width = max(num_vision_latent_frames)
        timesteps = ts.new_zeros(batch_size, width)
        sigmas = sg.new_zeros(batch_size, width)
        for i, lay in enumerate(layouts):
            r, c, _ = lay.video_metadata(device=sg.device)
            timesteps[i, : len(r)] = torch.where(r == CONDITION_VIDEO, 0, ts[i, c])
            sigmas[i, : len(r)] = torch.where(r == CONDITION_VIDEO, 0, sg[i, c])
        self._ar_step.chunk_ids = torch.arange(n, device=sg.device)
        return timesteps, sigmas

    def _add_noise_to_input(
        self,
        gen_data_clean,
        packed_sequence,
        sigmas,
        sigmas_action=None,
        sigmas_sound=None,
        sigmas_lidar=None,
        iteration=None,
    ):
        step = self._ar_step
        if step is None or step.action_sigmas is None:
            raise RuntimeError("independent chunk sigmas required")
        rows = []
        for i, lay in enumerate(self._joint_layouts):
            r, c, _ = lay.action_metadata(device=step.action_sigmas.device)
            rows.append(torch.where(r == STATE, 0, step.action_sigmas[i, c]))
        result = OmniMoTCausalModel._add_noise_to_input(
            self,
            gen_data_clean,
            packed_sequence,
            sigmas,
            sigmas_action=rows,
            sigmas_sound=sigmas_sound,
            sigmas_lidar=sigmas_lidar,
            iteration=iteration,
        )
        tmax = float(self.rectified_flow_action.noise_scheduler.config.num_train_timesteps)
        step.action_timesteps = [x * tmax for x in rows]
        packed_sequence.action.timesteps = (
            torch.cat(
                [
                    step.action_timesteps[i][idx.to(rows[i].device)]
                    for i, idx in enumerate(packed_sequence.action.noisy_frame_indexes)
                ]
            )
            .float()
            .cpu()
        )
        packed_sequence.uses_single_timestep = False
        return result

    def _validate_teacher_forcing_pack(self, packed_seq):
        # Explicit multi-sample layout replaces the stock control/target shape inference.
        if getattr(packed_seq, "joint_layout_version", None) != LAYOUT_VERSION:
            raise ValueError("teacher forcing requires declared joint layout")
        layouts = packed_seq.joint_layouts
        if len(layouts) != len(packed_seq.sample_lens):
            raise ValueError("sample ownership mismatch")
        if packed_seq.action is None or packed_seq.vision is None:
            raise ValueError("V/A payloads required")
        if sum(x.num_tokens for x in layouts) != sum(
            packed_seq.vision_item_split_lens[i][0] for i in range(len(layouts))
        ):
            raise ValueError("GEN ownership mismatch")

    def _build_tf_memory_state(
        self, packed_sequence, memory_info, net=None, detach_clean_kv=None, selected_clean_gen_token_indexes=None
    ):
        memory = OmniMoTCausalModel._build_tf_memory_state(
            self,
            packed_sequence,
            memory_info,
            net=net,
            detach_clean_kv=detach_clean_kv,
            selected_clean_gen_token_indexes=selected_clean_gen_token_indexes,
        )
        layouts = getattr(packed_sequence, "joint_layouts", self._joint_layouts)
        attention = JointTeacherForcingAttention(
            layouts, device=packed_sequence.vision.tokens[0].device, text_lengths=packed_sequence.joint_text_lengths
        )
        # Wrappers live on memory itself: strong bound methods/closures would
        # retain all clean KV tensors until cyclic GC, across training updates.
        memory_ref = weakref.ref(memory)
        init_base = weakref.WeakMethod(memory.init)
        text_lengths = packed_sequence.joint_text_lengths

        def init(hidden_states, device):
            init_base()(hidden_states, device)
            # Stock replay treats all text as one caption. Keep each packed sample causal and isolated.
            memory_ref().und_kv_offsets = torch.tensor(
                [0] + list(__import__("itertools").accumulate(text_lengths)), device=device, dtype=torch.int32
            )

        memory.init = init
        read_base = weakref.WeakMethod(memory.read_for_layer)

        def read_for_layer(i):
            value = read_base()(i)
            value.gen_attention_override = attention
            return value

        memory.read_for_layer = read_for_layer
        memory._joint_clean_text_kv = {}
        write_base = weakref.WeakMethod(memory.write_for_layer)

        def write_for_layer(i, kv_to_store):
            write_base()(i, kv_to_store)
            owner = memory_ref()
            if owner.pass_number == 1:
                # Preserve the clean text computation graph; inference UndKVCache detaches.
                owner._joint_clean_text_kv[i] = tuple(x.clone() for x in kv_to_store[2:])

        memory.write_for_layer = write_for_layer
        return memory

    def denoise(self, net=None, data_batch_packed=None, memory=None, video_temporal_causal=None):
        from .ar_v02_compact import compact_joint_targets, restore_joint_predictions, CompactJointMemory

        compact = (
            getattr(memory, "pass_number", None) == 2
            and hasattr(memory, "_joint_clean_text_kv")
            and getattr(data_batch_packed, "joint_layout_version", None) == LAYOUT_VERSION
            and getattr(self, "compact_noisy_training", True)
        )
        if not compact:
            return super().denoise(net, data_batch_packed, memory, video_temporal_causal)
        original = data_batch_packed
        targets, rows = compact_joint_targets(original)
        targets.to_cuda()
        view = CompactJointMemory(
            memory, original.joint_layouts, original.joint_text_lengths, original.vision.tokens[0].device
        )
        out = super().denoise(net, targets, view, video_temporal_causal)
        return restore_joint_predictions(out, original, rows)
