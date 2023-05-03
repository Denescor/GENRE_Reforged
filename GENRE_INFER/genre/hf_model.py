# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List

import torch
from genre.utils import chunk_it
from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)


#class GENREHubInterface(BartForConditionalGeneration):



class GENRE(BartForConditionalGeneration):
    #def __init__(self, model_name_or_path):
    #    super().__init__(model_name_or_path)
    #    self.model = BartForConditionalGeneration(model_name_or_path)
    #    self.
    #    self.init_weights()
    #    return self.model
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, archive_map=None, local=None):
        #unused args archive_map & local. Used for fairseq compatibility
        model = super().from_pretrained(model_name_or_path)
        model.tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        return model

    def sample(
        self, sentences: List[str], num_beams: int = 5, num_return_sequences=5, **kwargs
    ) -> List[str]:

        input_args = {
            k: v.to(self.device)
            for k, v in self.tokenizer.batch_encode_plus(
                sentences, padding=True, return_tensors="pt"
            ).items()
        }
        
        len_sentences = [len(sent) for sent in sentences]
        len_tokens = [len(sent[0]) for sent in input_args.values()]
        print("len sentences ({}) : {} / {}".format(len(len_sentences), min(len_sentences), max(len_sentences)))
        print("len tokens ({}) : {} / {}".format(len(len_tokens), min(len_tokens), max(len_tokens)))
        print("min gen tokens : {}\nmax gen tokens : {}".format(min(len_tokens)*2, min(len_tokens)*4))
        #print('input : ', input_args)
        #inputs = self.encode(sentences[0])
        
        #print("len sentences : {}".format(len(sentences)))
        print("inputs : {}".format(input_args))
        #print("inputs bis : {}".format(inputs))
        #print("inputs ters : {}".format(self.tokenizer.encode(sentences[0])))
        #print("inputs decode : {}".format(self.tokenizer.decode(inputs)))
        
        print(len_tokens)
        
        outputs = self.generate(
            **input_args,
            #inputs["inputs_ids"],
            min_length=min(len_tokens)*2,
            max_length=min(len_tokens)*4,
            #min_new_tokens=50, #set this argument if available 
            #max_new_tokens=500,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )

        len_out_tokens = [len(sent) for sent in outputs.sequences]
        len_out_sentences = [len(sent) for sent in self.tokenizer.batch_decode(
                             outputs.sequences, skip_special_tokens=True)]

        print("len outputs tokens ({}) : {} / {}".format(len(len_out_tokens), min(len_out_tokens), max(len_out_tokens)))
        print("len outputs sentences ({}) : {} / {}".format(len(len_out_sentences), min(len_out_sentences), max(len_out_sentences)))
        #return [[{"text": self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False), "logprob" : outputs.sequences_scores, "tensor" : outputs}]]

        return chunk_it(
            [
                {
                    "text": text,
                    "logprob": score,
                }
                for text, score in zip(
                    self.tokenizer.batch_decode(
                        outputs.sequences, skip_special_tokens=True
                    ),
                    outputs.sequences_scores,
                )
            ],
            num_return_sequences,
        )

    def encode(self, sentence):
        return self.tokenizer.encode(sentence, return_tensors="pt")[0]