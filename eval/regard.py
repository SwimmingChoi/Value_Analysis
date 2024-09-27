# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Iterator, List, Optional, Iterable

import torch
import argparse
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer

@dataclass
class Registration:
    name: str
    factory: Callable
    add_args: Optional[Callable] = None


class Registry:
    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable,
        add_args: Optional[Callable] = None,
    ):
        if not hasattr(cls, "registry"):
            cls.registry = {}
        cls.registry[name] = Registration(
            name=name,
            factory=factory,
            add_args=add_args,
        )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        assert hasattr(cls, "registry")
        for _, registration in cls.registry.items():
            if registration.add_args is not None:
                registration.add_args(parser)
        return parser

    @classmethod
    def build(cls, name, args):
        assert hasattr(cls, "registry")
        assert name in cls.registry, name
        return cls.registry[name].factory(args)


def batch_iter(it: Iterable, batch_size: int):
    batch = []

    for item in it:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch.clear()

    if batch:
        yield batch


@dataclass
class MetricConfig:
    batch_size: int = 1

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--metric-batch-size", type=int, default=1, required=False)
        return parser

    @classmethod
    def from_args(cls, args):
        return MetricConfig(
            batch_size=args.metric_batch_size,
        )


@dataclass
class Score:
    score: float
    label: str
    prompt: str
    result_data: str
    meta: Dict[str, Any]


@dataclass
class MetricResult:
    scores: List[Score]
    stats: Dict[str, Any]


class Metric(Registry):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--metric", type=str, required=True)
        parser = MetricConfig.add_args(parser)
        return parser

    @classmethod
    def build(cls, args):
        return super().build(args.metric, args)

    @property
    def name(self):
        raise NotImplementedError()

    def score(self, _: Iterator) -> MetricResult:
        raise NotImplementedError()


class HFClassifierMetric(Metric):
    @classmethod
    def from_args(cls, args, **kwargs):
        return HFClassifierMetric(
            name=args.metric,
            config=MetricConfig.from_args(args),
            **kwargs,
        )

    def __init__(
        self,
        name: str = "",
        model_id: str = "",
        labels: List[str] = [],
        config: MetricConfig = None,
        result_data = None,
        summarize: Optional[Callable[[List[Score]], Dict[str, Any]]] = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._name = name
        self.model_id = model_id
        self.labels = labels
        self.config = config
        self.batch_size = config.batch_size
        self.device = device
        self.result_data = result_data
        self.summarize = summarize

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        if not hasattr(self, "_model"):
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            model = model.eval().to(self.device)
            self._model = model
        return self._model

    @property
    def tokenizer(self):
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer


    def _summarize(self, scores: List[Score]) -> Dict[str, Any]:
        if self.summarize is None:
            return {}
        return self.summarize(scores)

    def score(self, result_data: Iterator) -> MetricResult:
        scores = []
        for batch in batch_iter(result_data, self.batch_size):
            inputs = self.tokenizer(
                [p.generation for p in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
            outputs = self.model.forward(**inputs)
            conf = torch.softmax(outputs.logits.cpu(), dim=-1)
            batch_scores, batch_labels = torch.topk(conf, dim=-1, k=1)
            for i in range(len(batch)):
                scores.append(
                    Score(
                        prompt=batch[i].prompt,
                        prediction=batch[i].generation,
                        score=batch_scores[i].item(),
                        label=self.labels[batch_labels[i].item()],
                        meta=batch[i].meta,
                    )
                )

        stats = self._summarize(scores)
        return MetricResult(scores=scores, stats=stats)

