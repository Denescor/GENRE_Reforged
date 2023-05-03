# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os
from collections import defaultdict
from typing import Dict, List

import torch
#from fairseq import search, utils
from fairseq.models.bart import BARTHubInterface, BARTModel
from omegaconf import open_dict

logger = logging.getLogger(__name__)


class GENREHubInterface(BARTHubInterface):
    def sample(
        self,
        sentences: List[str],
        beam: int = 5,
        verbose: bool = False,
        text_to_id=None,
        marginalize=False,
        marginalize_lenpen=0.5,
        max_len_a=1024,
        max_len_b=1024,
        skip_invalid_size_inputs=True,
        only_first = False,
        **kwargs,
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]

        #print(tokenized_sentences)

        batched_hypos = self.generate(
            tokenized_sentences,
            beam,
            verbose,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            skip_invalid_size_inputs=skip_invalid_size_inputs,
            **kwargs,
        )
        if only_first: 
            if len(batched_hypos) > 0 and len(batched_hypos[0]) > 0:
                outputs = [[{"text": self.decode(batched_hypos[0][0]["tokens"]), "score": batched_hypos[0][0]["score"]}]]
            else: outputs = []
        else:
            outputs = [
                [
                    {"text": self.decode(hypo["tokens"]), "score": hypo["score"]}
                    for hypo in hypos
                ]
                for hypos in batched_hypos
            ]
        if text_to_id: outputs = self.apply_text_to_id(batched_hypos, outputs, text_to_id, marginalize, marginalize_lenpen)
        return outputs

    def generate(self, *args, **kwargs) -> List[List[Dict[str, torch.Tensor]]]:
        return super(BARTHubInterface, self).generate(*args, **kwargs)

    def encode(self, sentence) -> torch.LongTensor:
        tokens = super(BARTHubInterface, self).encode(sentence)
        tokens[
            tokens >= len(self.task.target_dictionary)
        ] = self.task.target_dictionary.unk_index
        if tokens[0] != self.task.target_dictionary.bos_index:
            return torch.cat(
                (torch.tensor([self.task.target_dictionary.bos_index]), tokens)
            )
        else:
            return tokens
        
    def apply_text_to_id(batched_hypos, outputs, text_to_id, marginalize, marginalize_lenpen):
        outputs = [
            [{**hypo, "id": text_to_id(hypo["text"])} for hypo in hypos]
            for hypos in outputs
        ]

        if marginalize:
            for (i, hypos), hypos_tok in zip(enumerate(outputs), batched_hypos):
                outputs_dict = defaultdict(list)
                for hypo, hypo_tok in zip(hypos, hypos_tok):
                    outputs_dict[hypo["id"]].append(
                        {**hypo, "len": len(hypo_tok["tokens"])}
                    )

                outputs[i] = sorted(
                    [
                        {
                            "id": _id,
                            "texts": [hypo["text"] for hypo in hypos],
                            "scores": torch.stack(
                                [hypo["score"] for hypo in hypos]
                            ),
                            "score": torch.stack(
                                [
                                    hypo["score"]
                                    * hypo["len"]
                                    / (hypo["len"] ** marginalize_lenpen)
                                    for hypo in hypos
                                ]
                            ).logsumexp(-1),
                        }
                        for _id, hypos in outputs_dict.items()
                    ],
                    key=lambda x: x["score"],
                    reverse=True,
                )
        return outputs

    
class GENRE(BARTModel):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        archive_map=None,
        local="https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe",
        **kwargs,
    ):
        from fairseq import hub_utils

        if archive_map is None: archive_map=cls.hub_models()

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=archive_map,
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )

        local_bpe = {'_name': 'gpt2', 'gpt2_encoder_json': '{}/encoder.json'.format(local), 'gpt2_vocab_bpe': '{}/vocab.bpe'.format(local)}
        if local_bpe != x["args"]["bpe"]: 
            x["args"]["bpe"]["gpt2_encoder_json"] = local_bpe["gpt2_encoder_json"]
            x["args"]["bpe"]["gpt2_vocab_bpe"] = local_bpe["gpt2_vocab_bpe"]
        
        logging.debug("----------------------------------------")
        logging.debug("args : {}".format(x["args"]))
        logging.debug("task : {}".format(x["task"]))
        logging.debug("models : {}".format(x["models"]))
        logging.debug("----------------------------------------")
        logging.debug("bpe : {}".format(x["args"]["bpe"]))
        logging.debug("----------------------------------------")
        
        return GENREHubInterface(x["args"], x["task"], x["models"][0])


class mGENRE(BARTModel):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        sentencepiece_model="spm_256000.model",
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="sentencepiece",
        layernorm_embedding=True,
        archive_map=None,
        local="https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe",
        **kwargs,
    ):
        from fairseq import hub_utils

        if archive_map is None: archive_map=cls.hub_models()

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=archive_map,
            bpe=bpe,
            load_checkpoint_heads=True,
            sentencepiece_model=os.path.join(local, sentencepiece_model),
            sentencepiece_vocab=os.path.join(local, sentencepiece_model),
            **kwargs,
        )
        
        print(x["args"].keys())
        try: print(x["args"]["bpe"])
        except KeyError: print("not 'bpe' in args")
        try: print(x["args"]["sentencepiece"])
        except KeyError: print("not 'sentencepiece' in args")
        
        #local_bpe = {'_name': 'gpt2', 'gpt2_encoder_json': '{}/encoder.json'.format(local), 'gpt2_vocab_bpe': '{}/vocab.bpe'.format(local)}
        #if local_bpe != x["args"]["bpe"]: 
        #    x["args"]["bpe"]["gpt2_encoder_json"] = local_bpe["gpt2_encoder_json"]
        #    x["args"]["bpe"]["gpt2_vocab_bpe"] = local_bpe["gpt2_vocab_bpe"]
            
        return GENREHubInterface(x["args"], x["task"], x["models"][0])
    
class BARThez(BARTModel):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        sentencepiece_model="sentence.bpe.model",
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="sentencepiece",
        task="translation",
        layernorm_embedding=True,
        archive_map=None,
        local="https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe",
        **kwargs,
    ):
        from fairseq import hub_utils

        if archive_map is None: archive_map=cls.hub_models()

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=archive_map,
            bpe=bpe,
            load_checkpoint_heads=True,
            sentencepiece_model=os.path.join(local, sentencepiece_model),
            sentencepiece_vocab=os.path.join(local, sentencepiece_model),
            **kwargs,
        )
        
        print("task : {}".format(x["task"]))
        try: print(x["args"].keys())
        except: print(type(x["args"]))
        else:
            try: print(x["args"]["bpe"])
            except KeyError: print("not 'bpe' in args")
            try: print(x["args"]["sentencepiece"])
            except KeyError: print("not 'sentencepiece' in args")
        
        #local_bpe = {'_name': 'gpt2', 'gpt2_encoder_json': '{}/encoder.json'.format(local), 'gpt2_vocab_bpe': '{}/vocab.bpe'.format(local)}
        #if local_bpe != x["args"]["bpe"]: 
        #    x["args"]["bpe"]["gpt2_encoder_json"] = local_bpe["gpt2_encoder_json"]
        #    x["args"]["bpe"]["gpt2_vocab_bpe"] = local_bpe["gpt2_vocab_bpe"]
            
        return GENREHubInterface(x["args"], x["task"], x["models"][0])
