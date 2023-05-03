import argparse
import json
import urllib.parse
import pickle
import tqdm


def create_label_json(begin, end, qid):
    return {
        "span": [begin, end],
        "recognized_by": "GENRE",
        "id": qid,
        "linked_by": "GENRE",
        "candidates": [qid]
    }


def compute_labels_OLD(paragraph: str, labeled_paragraph: str, start_position: int):
    # on retire les doubles espaces
    paragraph = " ".join(paragraph.split()).strip()
    labeled_paragraph = " ".join(labeled_paragraph.split()).strip()
    if paragraph == labeled_paragraph:
        return [], False
    p_pos = 0 #char position in initial text (paragraph)
    l_pos = 0 #char position in labeled text (labeled_paragraph)
    start = end = 0 #position of mentions
    labels = []#list of mentions/entities
    cut = False #incomplete computing of labels
    while p_pos < len(paragraph):
        p_char = paragraph[p_pos] #char at position 'p_pos' in initial text
        if p_char in " []\n":
            p_pos += 1
            continue
        l_char = labeled_paragraph[l_pos] #char at position 'l_pos' in labeled text
        if l_char in " \n":
            l_pos += 1
        elif l_char == p_char:
            l_pos += 1
            p_pos += 1
            end = p_pos
        elif l_char == "{":
            start = p_pos
            l_pos += 1
        elif l_char == "}":
            label_start = l_pos + 3
            label_end = labeled_paragraph.find("]", label_start)
            label = labeled_paragraph[label_start:label_end].strip()
            l_pos = label_end + 1
            labels.append((start_position + start, start_position + end, label))
        else:
            #print("paragraph : \n'{}'".format(labeled_paragraph))
            #print("- l_char : '{}' ({})\n- p_char : '{}' ({})".format(l_char, labeled_paragraph[l_pos-5:l_pos+5], p_char, paragraph[p_pos-5:p_pos+5]))
            #print("- label : '{}' ({} - {})".format(labels[-1], label_start, label_end))
            cut = True
            break
    return labels, cut

def compute_labels(paragraphe: str, labeled_paragraphe: str, start_position: int):
    """
    inefficace
    """
    paragraph = " ".join(paragraphe.split()).strip()
    labeled_paragraph = " ".join(labeled_paragraphe.split()).strip()
    for s_char in "-_~":
        paragraph = s_char.join([x.strip() for x in paragraph.split(s_char)])
        labeled_paragraph = s_char.join([x.strip() for x in labeled_paragraph.split(s_char)])
    if paragraph == labeled_paragraph:
        return [], False
    p_pos = 0 #char position in initial text (paragraph)
    l_pos = 0 #char position in labeled text (labeled_paragraph)
    is_mention, is_label = False, False
    mentions = []
    start = end = -1 #position of mentions
    labels = [] #list of mentions/entities
    cut = False #incomplete computing of labels
    while p_pos < len(paragraph):
        #if l_pos >= len(labeled_paragraph):
        #    print("paragraph : \n'{}'".format(labeled_paragraph))
        #    print("Désynchronisation fin paragraph:")
        #    print("- l_char : '{}'\n- p_char : '{}'".format(l_char, p_char))
        #    print("- mention : '{}' ({} - {})".format(mentions[-1], mention_start, mention_end))
        #    print("- label : '{}' ({} - {})".format(labels[-1], label_start, label_end))
        p_char = paragraph[p_pos] #char at position 'p_pos' in initial text
        l_char = labeled_paragraph[l_pos] #char at position 'l_pos' in labeled text
        if p_char in "[]\n": p_pos += 1
        elif p_char == l_char and (not is_label): #same text
            p_pos += 1
            l_pos += 1
            if (p_char in ".()[]{}") and (p_pos+1 < len(paragraph)) and (paragraph[p_pos+1] != labeled_paragraph[l_pos+1]):
                #print("p_char : '{}'\nl_char : '{}'".format(paragraph[p_pos:p_pos+2], labeled_paragraph[l_pos:l_pos+2]))
                l_pos += 1
        elif l_char == "{" and (not is_mention): #begining of mention
            is_mention = True
            start = p_pos #start of mention in initial text
            mention_start = l_pos + 2
            l_pos += 1
        elif l_char == "}" and is_mention:
            mention_end = l_pos - 1
            end = p_pos #end of mention in initial text
            l_pos += 1
        elif l_char == "[" and is_mention:
            is_label = True
            is_mention = False
            label_start = l_pos + 2
            l_pos += 1
        elif l_char == "]" and is_label:
            label_end = l_pos - 1
            if labeled_paragraph[l_pos+2] != paragraph[p_pos]:
                l_pos += 1
            else:
                l_pos += 2
            is_label = False
            mention = labeled_paragraph[mention_start:mention_end].strip()
            mentions.append(mention)
            label = labeled_paragraph[label_start:label_end].strip()
            labels.append((start_position + start, start_position + end, label))
            start, end, label_start, label_end, mention_start, mention_end = -1, -1, -1, -1, -1, -1
        elif is_mention or is_label:
            l_pos += 1
        
        else:
            if p_char in "-_.~":
                l_pos += 1
                continue
            #print("paragraph : \n'{}'".format(labeled_paragraph))
            #if is_mention or is_label:
            #    print("Désynchronisation dans mention:")
            #else:
            #    print("Désynchronisation hors mention:")
            #print("- l_char : '{}' ({})\n- p_char : '{}' ({})".format(l_char, labeled_paragraph[l_pos-5:l_pos+5], p_char, paragraph[p_pos-5:p_pos+5]))
            #print("- mention : '{}' ({} - {})".format(mentions[-1], mention_start, mention_end))
            #print("- label : '{}' ({} - {})".format(labels[-1], label_start, label_end))
            cut = True
            break
    return labels, cut
        
            
            
            


def get_mapping(lang):
    prefix = "https://{}.wikipedia.org/wiki/".format(lang)
    mapping = {}
    for line in open("data/elevant/{}/qid_to_wikipedia_url.tsv".format(lang)):
        line = line[:-1]
        vals = line.split("\t")
        qid = vals[0]
        wikipedia_title = urllib.parse.unquote(vals[1][len(prefix):]).replace("_", " ")
        mapping[wikipedia_title] = qid
    return mapping


def main(args):
    
    if not args.wikipedia:
        print("read mapping...")
        mapping = get_mapping(args.lang)

        print("load redirects...")
        with open("data/elevant/{}/link_redirects.pkl".format(args.lang), "rb") as f:
            redirects = pickle.load(f)

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = args.input_file[:-4] + ".jsonl"

    miss = 0
    total = 0
    no_mention = 0
    incomp_mention = 0
    label_error = 0
    articles = []

    for line in open(args.input_file):
        articles.append(json.loads(line))
    total_articles = len(articles)
    
    with open(output_file, "w") as out_file:
        for data in tqdm.tqdm(articles, total=len(articles), desc="Transform Predictions"):
            #print("== " + str(data["id"]) + " (" + str(data["evaluation_span"]) + ") ==")
            text = data["text"]
            genre_text = data["GENRE"]
            position = 0
    
            labels = []
            try: wikipedia_labels, cut = compute_labels_OLD(text, genre_text, position)
            except Exception:
                label_error += 1
                continue
            total += len(wikipedia_labels)
            if len(wikipedia_labels) == 0: no_mention += 1
            if cut: incomp_mention += 1
            for start, end, label in wikipedia_labels:
                qid = label
                if args.wikipedia:
                    qid = "https://{}.wikipedia.org/wiki/".format(args.lang) + label.replace(" ", "_")
                else:
                    if label in mapping:
                        qid = mapping[label]
                    elif label in redirects:
                        redirected = redirects[label]
                        if redirected in mapping:
                            qid = mapping[redirected]
                    else: miss += 1
                #print(start, end, label, qid)
                labels.append(create_label_json(start, end, qid))
            data["entity_mentions"] = labels
            out_file.write(json.dumps(data) + "\n")
    print("articles unable to compute : {}/{} ({:.2f}%)".format(label_error, total_articles, 100*(label_error/total_articles)))
    print("articles without mentions : {}/{} ({:.2f}%)".format(no_mention, total_articles, 100*(no_mention/total_articles)))
    if total_articles > no_mention + label_error:
        print("articles partially compute : {}/{} ({:.2f}%)".format(incomp_mention, total_articles-no_mention, 100*(incomp_mention/(total_articles-no_mention))))
        print("missed mentions : {}/{} ({:.2f}%)".format(miss, total, 100*(miss/total)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_file", type=str)
    parser.add_argument("-o", dest="output_file", type=str, default=None)
    parser.add_argument("-l", dest="lang", type=str, default="en")
    parser.add_argument("--wikipedia", action="store_true")
    args = parser.parse_args()
    print("compute for {} data".format(args.lang))
    main(args)
