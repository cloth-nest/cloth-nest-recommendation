from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import json
import torch
import pickle
import h5py
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable


def default_image_loader(path):
    return Image.open(path).convert("RGB")


def parse_iminfo(question, itemIdToIndex, itemIdentifierToItemId, gt=None):
    """Maps the questions from the FITB and compatibility tasks back to
    their index in the precomputed matrix of features

    question: List of images to measure compatibility between
    im2index: Dictionary mapping an image name to its location in a
              precomputed matrix of features
    gt: optional, the ground truth outfit set this item belongs to
    """
    questions = []
    is_correct = np.zeros(len(question), np.bool_)
    for index, itemIdentifier in enumerate(question):
        set_id = itemIdentifier.split("_")[0]
        if gt is None:
            gt = set_id

        itemId = itemIdentifierToItemId[itemIdentifier]
        questions.append((itemIdToIndex[itemId], itemId))
        is_correct[index] = set_id == gt

    return questions, is_correct, gt


def load_typespaces(rootdir, rand_typespaces, num_rand_embed):
    """loads a mapping of pairs of types to the embedding used to
    compare them

    rand_typespaces: Boolean indicator of randomly assigning type
                     specific spaces to their embedding
    num_rand_embed: number of embeddings to use when
                    rand_typespaces is true
    """
    typespace_fn = os.path.join(rootdir, "typespaces.p")
    typespaces = pickle.load(open(typespace_fn, "rb"))
    if not rand_typespaces:
        ts = {}
        for index, t in enumerate(typespaces):
            ts[t] = index

        typespaces = ts
        return typespaces

    # load a previously created random typespace or create one
    # if none exist
    width = 0
    fn = os.path.join(rootdir, "typespaces_rand_%i.p") % num_rand_embed
    if os.path.isfile(fn):
        typespaces = pickle.load(open(fn, "rb"))
    else:
        spaces = np.random.permutation(len(typespaces))
        width = np.ceil(len(spaces) / float(num_rand_embed))
        ts = {}
        for index, t in enumerate(spaces):
            ts[typespaces[t]] = int(np.floor(index / width))

        typespaces = ts
        pickle.dump(typespaces, open(fn, "wb"))

    return typespaces


def load_compatibility_questions(fn, itemIdToIndex, itemIdentifierToItemId):
    """Returns the list of compatibility questions for the
    split"""
    with open(fn, "r") as f:
        lines = f.readlines()

    compatibility_questions = []
    for line in lines:
        data = line.strip().split()
        # [1:] means taking items from index 1 -> end in list
        compat_question, _, _ = parse_iminfo(data[1:], itemIdToIndex, itemIdentifierToItemId)
        compatibility_questions.append((compat_question, int(data[0])))

    return compatibility_questions


def load_fitb_questions(fn, im2index, id2im):
    """Returns the list of fill in the blank questions for the
    split"""
    data = json.load(open(fn, "r"))
    questions = []
    for item in data:
        question = item["question"]
        q_index, _, gt = parse_iminfo(question, im2index, id2im)
        answer = item["answers"]
        a_index, is_correct, _ = parse_iminfo(answer, im2index, id2im, gt)
        questions.append((q_index, a_index, is_correct))

    return questions


class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        split,
        meta_data,
        text_dim=None,
        transform=None,
        loader=default_image_loader,
    ):
        #region Setting directories to images and data json file
        rootdir = os.path.join(args.datadir, "polyvore_outfits", args.polyvore_split)
        """Root of the polyvore split (disjoint/non-disjoit)"""
        
        self.imagePath = os.path.join(args.datadir, "polyvore_outfits", "images")
        self.is_train = split == "train"
        
        
        data_json = os.path.join(rootdir, "%s.json" % split)
        """JSON File for phase (train, val, test)"""

        
        outfit_data = json.load(open(data_json, "r"))
        #endregion

        # get list of images and make a mapping used to quickly organize the data
        itemId2Category = {}
        category2ims = {}
        imageNames = set()
        itemIdentifier2ItemId = {}
        for outfit in outfit_data:
            outfit_id = outfit["set_id"]
            for item in outfit["items"]:
                itemId = item["item_id"]
                category = meta_data[itemId]["semantic_category"]
                itemId2Category[itemId] = category

                if category not in category2ims:
                    category2ims[category] = {}

                if outfit_id not in category2ims[category]:
                    category2ims[category][outfit_id] = []

                category2ims[category][outfit_id].append(itemId)
                itemIdentifier2ItemId["%s_%i" % (outfit_id, item["index"])] = itemId
                imageNames.add(itemId)

        imageNames = list(imageNames)
        itemIdToIndex = {}
        for index, itemId in enumerate(imageNames):
            itemIdToIndex[itemId] = index

        self.data = outfit_data
        self.imageNames = imageNames
        self.itemId2Category = itemId2Category
        self.typespaces = load_typespaces(
            rootdir, args.rand_typespaces, args.num_rand_embed
        )
        self.transform = transform
        self.loader = loader
        self.split = split

        if self.is_train:
            self.text_feat_dim = text_dim
            self.desc2vecs = {}
            featfile = os.path.join(rootdir, "train_hglmm_pca6000.txt")
            with open(featfile, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    vec = line.split(",")
                    label = ",".join(vec[: -self.text_feat_dim])
                    vec = np.array(
                        [float(x) for x in vec[-self.text_feat_dim :]], np.float32
                    )
                    assert len(vec) == text_dim
                    self.desc2vecs[label] = vec

            self.itemIdToDescription = {}
            for itemId in imageNames:
                desc = meta_data[itemId]["title"]
                if not desc:
                    desc = meta_data[itemId]["url_name"]

                desc = desc.replace("\n", "").encode("ascii", "ignore").strip().lower()

                # sometimes descriptions didn't map to any known words so they were
                # removed, so only add those which have a valid feature representation
                if desc and desc in self.desc2vecs:
                    self.itemIdToDescription[itemId] = desc

            # At train time we pull the list of outfits and enumerate the pairwise
            # comparisons between them to train with.  Negatives are pulled by the
            # __get_item__ function
            pos_pairs = []
            max_items = 0
            for outfit in outfit_data:
                items = outfit["items"]
                cnt = len(items)
                max_items = max(cnt, max_items)
                outfit_id = outfit["set_id"]
                for j in range(cnt - 1):
                    for k in range(j + 1, cnt):
                        pos_pairs.append(
                            [outfit_id, items[j]["item_id"], items[k]["item_id"]]
                        )

            self.pos_pairs = pos_pairs
            self.category2ims = category2ims
            self.max_items = max_items
        else:
            # pull the two task's questions for test and val splits
            fn = os.path.join(rootdir, "fill_in_blank_%s.json" % split)
            self.fitb_questions = load_fitb_questions(fn, itemIdToIndex, itemIdentifier2ItemId)
            fn = os.path.join(rootdir, "compatibility_%s.txt" % split)
            self.compatibility_questions = load_compatibility_questions(
                fn, itemIdToIndex, itemIdentifier2ItemId
            )

    def load_train_item(self, image_id):
        """Returns a single item in the triplet and its data"""
        imfn = os.path.join(self.imagePath, "%s.jpg" % image_id)
        img = self.loader(imfn)
        if self.transform is not None:
            img = self.transform(img)

        if image_id in self.itemIdToDescription:
            text = self.itemIdToDescription[image_id]
            text_features = self.desc2vecs[text]
            has_text = 1
        else:
            text_features = np.zeros(self.text_feat_dim, np.float32)
            has_text = 0.0

        has_text = np.float32(has_text)
        item_type = self.itemId2Category[image_id]
        return img, text_features, has_text, item_type

    def sample_negative(self, outfit_id, item_id, item_type):
        """Returns a randomly sampled item from a different set
        than the outfit at data_index, but of the same type as
        item_type

        data_index: index in self.data where the positive pair
                    of items was pulled from
        item_type: the coarse type of the item that the item
                   that was paired with the anchor
        """
        item_out = item_id
        candidate_sets = self.category2ims[item_type].keys()
        attempts = 0
        while item_out == item_id and attempts < 100:
            choice = np.random.choice(candidate_sets)
            items = self.category2ims[item_type][choice]
            item_index = np.random.choice(range(len(items)))
            item_out = items[item_index]
            attempts += 1

        return item_out

    def get_typespace(self, anchor, pair):
        """Returns the index of the type specific embedding
        for the pair of item types provided as input
        """
        query = (anchor, pair)
        if query not in self.typespaces:
            query = (pair, anchor)

        return self.typespaces[query]

    def test_compatibility(self, embeds, metric):
        """Returns the area under a roc curve for the compatibility
        task

        embeds: precomputed embedding features used to score
                each compatibility question
        metric: a function used to score the elementwise product
                of a pair of embeddings, if None euclidean
                distance is used
        """
        scores = []
        labels = np.zeros(len(self.compatibility_questions), np.int32)
        for index, (outfit, label) in enumerate(self.compatibility_questions):
            labels[index] = label
            n_items = len(outfit)
            outfit_score = 0.0
            num_comparisons = 0.0
            for i in range(n_items - 1):
                item1, img1 = outfit[i]
                type1 = self.itemId2Category[img1]
                for j in range(i + 1, n_items):
                    item2, img2 = outfit[j]
                    type2 = self.itemId2Category[img2]
                    condition = self.get_typespace(type1, type2)
                    embed1 = embeds[item1][condition].unsqueeze(0)
                    embed2 = embeds[item2][condition].unsqueeze(0)
                    if metric is None:
                        outfit_score += torch.nn.functional.pairwise_distance(
                            embed1, embed2, 2
                        )
                    else:
                        outfit_score += metric(Variable(embed1 * embed2)).data

                    num_comparisons += 1.0

            outfit_score /= num_comparisons
            scores.append(outfit_score)

        scores = torch.cat(scores).squeeze().cpu().numpy()
        # scores = np.load('feats.npy')
        # print(scores)
        # assert(False)
        # np.save('feats.npy', scores)
        auc = roc_auc_score(labels, 1 - scores)
        return auc

    def test_fitb(self, embeds, metric):
        """Returns the accuracy of the fill in the blank task

        embeds: precomputed embedding features used to score
                each compatibility question
        metric: a function used to score the elementwise product
                of a pair of embeddings, if None euclidean
                distance is used
        """
        correct = 0.0
        n_questions = 0.0
        for q_index, (questions, answers, is_correct) in enumerate(self.fitb_questions):
            answer_score = np.zeros(len(answers), dtype=np.float32)
            for index, (answer, img1) in enumerate(answers):
                type1 = self.itemId2Category[img1]
                score = 0.0
                for question, img2 in questions:
                    type2 = self.itemId2Category[img2]
                    condition = self.get_typespace(type1, type2)
                    embed1 = embeds[question][condition].unsqueeze(0)
                    embed2 = embeds[answer][condition].unsqueeze(0)
                    if metric is None:
                        score += torch.nn.functional.pairwise_distance(
                            embed1, embed2, 2
                        )
                    else:
                        score += metric(Variable(embed1 * embed2)).data

                answer_score[index] = score.squeeze().cpu().numpy()

            correct += is_correct[np.argmin(answer_score)]
            n_questions += 1

        # scores are based on distances so need to convert them so higher is better
        acc = correct / n_questions
        return acc

    def __getitem__(self, index):
        if self.is_train:
            outfit_id, anchor_im, pos_im = self.pos_pairs[index]
            img1, desc1, has_text1, anchor_type = self.load_train_item(anchor_im)
            img2, desc2, has_text2, item_type = self.load_train_item(pos_im)

            neg_im = self.sample_negative(outfit_id, pos_im, item_type)
            img3, desc3, has_text3, _ = self.load_train_item(neg_im)
            condition = self.get_typespace(anchor_type, item_type)
            return (
                img1,
                desc1,
                has_text1,
                img2,
                desc2,
                has_text2,
                img3,
                desc3,
                has_text3,
                condition,
            )

        anchor = self.imageNames[index]
        img1 = self.loader(os.path.join(self.imagePath, "%s.jpg" % anchor))
        if self.transform is not None:
            img1 = self.transform(img1)

        return img1

    def shuffle(self):
        np.random.shuffle(self.pos_pairs)

    def __len__(self):
        if self.is_train:
            return len(self.pos_pairs)

        return len(self.imageNames)
