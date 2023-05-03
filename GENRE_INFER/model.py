from typing import List, Optional

import spacy
import torch.cuda

from genre.fairseq_model import GENRE, mGENRE, BARThez
from genre.trie import DummyTrieEntity, DummyTrieMention
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from genre.entity_linking import _get_end_to_end_prefix_allowed_tokens_fn as get_prefix_allowed_tokens_fn_
from helper_pickle import pickle_load


class Model:
    def __init__(self,
                 model_path: str,
                 type_model: str,
                 mention_trie: Optional[str],
                 mention_to_candidates_dict: Optional[str],
                 candidates_trie: Optional[str],
                 archive_map: Optional[str], #None,
                 local: Optional[str], #"https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe"):
                 spacy: Optional[str],
                 architecture="fairseq"):
        if model_path == "yago":
            model_name = "models/fairseq_e2e_entity_linking_aidayago"
        elif model_path == "wiki-abs":
            model_name = "models/fairseq_e2e_entity_linking_wiki_abs"
        else:
            model_name = model_path #other models
        print("model loaded : {}".format(model_name))
        self.model_name = type_model
        if type_model == "GENRE": self.model = GENRE.from_pretrained(model_name, archive_map=archive_map, local=local).eval()
        elif type_model == "mGENRE": self.model = mGENRE.from_pretrained(model_name, archive_map=archive_map, local=local).eval()
        elif type_model == "BARThez": self.model = BARThez.from_pretrained(model_name, archive_map=archive_map, local=local).eval()
        else:
            print("'{}' not recognized".format(type_model))
            exit(0)
        if torch.cuda.is_available():
            print("move model to GPU...")
            self.model = self.model.cuda()
        #if type_model == "BARThez": self.model.half()
        self.mention_trie = pickle_load(mention_trie, verbose=True)
        self.mention_to_candidates_dict = pickle_load(mention_to_candidates_dict, verbose=True)
        self.candidates_trie = pickle_load(candidates_trie, verbose=True)
        self.spacy_model = spacy
        self._ensure_spacy() #force loading at init
        self._define_tries() #force loading at init
        
    def _define_static(self):
        self.start_mention_token = "{"
        self.end_mention_token = "}"
        self.start_entity_token = "["
        self.end_entity_token = "]"
        self.encode_fn = lambda x: self.model.encode(x).tolist()
        self.decode_fn = lambda x: self.model.decode(torch.tensor(x))
        self.bos_token_id = self.model.model.decoder.dictionary.bos()
        self.pad_token_id = self.model.model.decoder.dictionary.pad()
        self.eos_token_id = self.model.model.decoder.dictionary.eos()
        self.vocabulary_length = len(self.model.model.decoder.dictionary)
        
    def _define_tries(self):
        self._define_static()
        self.codes = {
            n: self.encode_fn(" {}".format(c))[1]
            for n, c in zip(
                (
                    "start_mention_token",
                    "end_mention_token",
                    "start_entity_token",
                    "end_entity_token",
                ),
                (
                    self.start_mention_token,
                    self.end_mention_token,
                    self.start_entity_token,
                    self.end_entity_token,
                ),
            )
        }
        self.codes["EOS"] = self.eos_token_id
    
        if self.mention_trie is None:
            self.mention_trie = DummyTrieMention(
                [
                    i for i in range(self.vocabulary_length)
                    if i not in (
                        self.bos_token_id,
                        self.pad_token_id,
                    )
                ]
            )
    
        if self.candidates_trie is None and self.mention_to_candidates_dict is None:
            self.candidates_trie = DummyTrieEntity(
                [
                    i for i in range(self.vocabulary_length)
                    if i not in (
                        self.bos_token_id,
                        self.pad_token_id,
                    )
                ],
                self.codes,
            )

    def _ensure_spacy(self):
        if self.spacy_model is None:
            self.spacy_model = spacy.load("en_core_web_sm")
        elif type(self.spacy_model) == str:
            self.spacy_model = spacy.load(self.spacy_model)

    def _split_sentences(self, text: str) -> List[str]:
        doc = self.spacy_model(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences

    def _split_long_texts(self, text: str) -> List[str]:
        MAX_WORDS = 150
        split_parts = []
        sentences = self._split_sentences(text)
        part = ""
        n_words = 0
        for sentence in sentences:
            sent_words = len(sentence.split())
            if len(part) > 0 and n_words + sent_words > MAX_WORDS:
                split_parts.append(part)
                part = ""
                n_words = 0
            if len(part) > 0:
                part += " "
            part += sentence
            n_words += sent_words
        if len(part) > 0:
            split_parts.append(part)
        return split_parts

    def predict_paragraphs(self, text: List[str], split_sentences: bool, split_long_texts: bool) -> str:
        sentences = []
        paragraph_id = []
        nb_error, total_sample = 0, 0
        for i, sent in enumerate(text):
            # On applatit text en 1 seule liste le découpage en paragraphes de tous les textes de la liste initiale
            split_sent = self.predict_paragraph(sent, split_sentences, split_long_texts)
            paragraph_id.append(len(split_sent))
            sentences.extend(split_sent)
            
        if len(sentences) == 0: return []
        
        predictions, error, total = self.predict(sentences) #on fait la prédiction d'un bloc
        total_sample += total
        nb_error += error
        
        if len(sentences) != len(predictions):
            print("missing predictions :\t- expectation : {}\t- reality : {}".format(len(sentences), len(predictions)))
            return text, 1, 1
        
        final_predictions = []
        i = 0
        try:
            for size in paragraph_id:
                #On recole les prédictions par paragraphes pour obtenir une liste similaire à celle initiale
                final_predictions.append(" ".join(predictions[i:i+size]))
                i += size
        except TypeError:
            print("decoding error. prediction found : {}".format(predictions[i]))
            return text, 1, 1
        if len(text) != len(final_predictions):
            print("bad final predictions :\t- expectation : {}\t- reality : {}".format(len(text), len(final_predictions)))
            return text, 1 ,1
        return final_predictions, nb_error, total_sample

    def predict_paragraph(self, text: str, split_sentences: bool, split_long_texts: bool) -> List[str]:
        if split_sentences:
            sentences = self._split_sentences(text)
        elif split_long_texts:
            sentences = self._split_long_texts(text)
        else:
            sentences = [text]
        return sentences
#        predictions = []
#        for sent in sentences: #, total=len(sentences), desc="Predict Paragraph"):
#            #print("IN:", sent)
#            if len(sent.strip()) == 0:
#                prediction = sent
#            else:
#                prediction = self.predict(sent)
#            #print("PREDICTION:", prediction)
#            predictions.append(prediction)
#        return " ".join(predictions)

    def predict_iteratively(self, text: str):
        text = self._preprocess(text)
        sentences = self._split_sentences(text)
        n_parts = 1
        while n_parts <= len(sentences):
            #plural_s = "s" if n_parts > 1 else ""
            #print(f"INFO: Predicting {n_parts} part{plural_s}.")
            sents_per_part = len(sentences) / n_parts
            results = []
            did_fail = False
            nb_error, total_sample = 0, 0
            for i in range(n_parts): #, total=n_parts, desc="Predict Iteratively {n_parts} part{plural_s}"):
                start = int(sents_per_part * i)
                end = int(sents_per_part * (i + 1))
                part = " ".join(sentences[start:end])
                #print("IN:", part)
                try:
                    result = self._query_model([part])[0]
                except Exception as e:
                    print(e)
                    nb_error += 1
                    result = None
                #else: print("RESULT:", result)
                finally: total_sample += 1
                if result is not None and len(result) > 0 and not _is_prediction_complete(part, result[0]["text"]):
                    nb_error += 1
                    results.append(part)
                    #print("inconsistent prediction")
                elif result is not None and len(result) > 0:
                    #print("is a prediction")
                    results.append(result[0]["text"])
                elif end - start == 1:
                    #print("is not a prediction")
                    results.append(part)
                else:
                    did_fail = True
                    break
            if did_fail:
                n_parts += 1
            else:
                return " ".join(results), nb_error, total_sample

    def _preprocess(self, text):
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text

    def _query_model(self, text):
        sentences = [" " + txt if (len(txt) > 0 and txt[0] != " ") else txt for txt in text]  # necessary to detect mentions in the beginning of a sentence
        sent_origs = [[self.codes["EOS"]] + self.encode_fn(sent)[1:] for sent in sentences]
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn_(
            self.encode_fn,
            self.decode_fn,
            self.bos_token_id,
            self.pad_token_id,
            self.eos_token_id,
            self.vocabulary_length,
            sentences, #unused if sent_origs is defined
            self.start_mention_token,
            self.end_mention_token,
            self.start_entity_token,
            self.end_entity_token,
            self.mention_trie,
            self.candidates_trie,
            self.mention_to_candidates_dict,
            self.codes,
            sent_origs #only parameters to recompute to every sentences
        )
        #prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
        #    self.model,
        #    sentences,
        #    mention_trie=self.mention_trie,
        #    mention_to_candidates_dict=self.mention_to_candidates_dict,
        #    candidates_trie=self.candidates_trie,
        #)
        if self.model_name == "BARThez": eval_barthez = dict(max_len_a=4096, max_len_b=4096, skip_invalid_size_inputs=True)
        else: eval_barthez = dict(max_len_a=1024, max_len_b=1024)
        result = self.model.sample(
            sentences,
            beam=5,
            verbose=False,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            **eval_barthez
            
        )
        return result

    def predict(self, text: List[str]) -> str:
        text = [self._preprocess(txt) for txt in text]

        try: result = self._query_model(text)
        except Exception as e: 
            print(e)
            return text, 1, 1 

        predictions = []
        nb_error, total_sample = 0, 0
        for i, res in enumerate(result):
            try: predictions.append(res[0]["text"].strip())
            except IndexError: 
                print(res)
                pred, error, total = self.predict_iteratively(text[i]) #This sentence cannot be predicted per_paragraph. So try iteratively
                predictions.append(pred)
                total_sample += total
                nb_error += error
            else: total_sample += 1
        
        #print(predictions)
        return predictions, nb_error, total_sample


def _is_prediction_complete(text, prediction):
    len_text = 0
    for char in text:
        if char != " ":
            len_text += 1
    len_prediction = 0
    inside_prediction = False
    for char in prediction:
        if char in " {}":
            continue
        elif char == "[":
            inside_prediction = True
        elif char == "]":
            inside_prediction = False
        elif not inside_prediction:
            len_prediction += 1
    return len_text == len_prediction
