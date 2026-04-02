"""Knowledge distillation for Phase 2 training.

Uses a teacher model (Qwen2.5-72B-Instruct) to guide the Controller's
learning. Combines CE loss with KL divergence from teacher logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> Tensor:
    """Compute combined distillation + CE loss.

    Loss = alpha * KL(softmax(teacher/T) || softmax(student/T)) * T^2
         + (1 - alpha) * CE(student, labels)

    Args:
        student_logits: Student model output. Shape [batch, seq, vocab].
        teacher_logits: Teacher model output. Shape [batch, seq, vocab].
        labels: Target token IDs. Shape [batch, seq].
        alpha: Weight for distillation loss (0 = pure CE, 1 = pure distill).
        temperature: Softmax temperature for distillation.

    Returns:
        Scalar loss tensor.
    """
    # Shift for next-token prediction
    shift_student = student_logits[..., :-1, :].contiguous()
    shift_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # CE loss
    ce_loss = F.cross_entropy(
        shift_student.view(-1, shift_student.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # KL divergence with temperature scaling
    student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)

    # Mask padding tokens
    mask = (shift_labels != -100).unsqueeze(-1).float()

    kl_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    )
    kl_loss = (kl_loss * mask).sum() / mask.sum()
    kl_loss = kl_loss * (temperature ** 2)

    return alpha * kl_loss + (1 - alpha) * ce_loss


class TeacherWrapper:
    """Wrapper for lazy-loading and caching teacher model outputs.

    The teacher model is large (72B) so we load it in 8-bit quantization
    and cache outputs to avoid repeated inference.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        device: str = "cuda",
        load_in_8bit: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load the teacher model."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"torch_dtype": torch.bfloat16}
        if self.load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.device

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **kwargs,
        )
        self._model.requires_grad_(False)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @torch.no_grad()
    def get_logits(self, input_ids: Tensor) -> Tensor:
        """Get teacher logits for given input IDs.

        Args:
            input_ids: Token IDs. Shape [batch, seq_len].

        Returns:
            Teacher logits. Shape [batch, seq_len, vocab].
        """
        self._load()
        outputs = self._model(input_ids.to(self._model.device))
        return outputs.logits.to(input_ids.device)
