import math
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
from src.data_loader.golden_standard_loader import GoldenStandardLoader
from transformers import BertTokenizer


class TableMultiColRandomIntersectStreamDataset(torch.utils.data.Dataset):
    # this tokenizer will be set in the function get_dataset
    tokenizer = None
    max_model_input_length = None

    def __init__(
        self,
        df,
        ds_length,
        frag_height,
        frag_width,
        table_name,
        static_n_sample=None,
        col_name_prob=0.0,
        col_name_variant_prob=0.5,
        schema_process_type=None,
        numerical_col_bins=0,
        numerical_col_window_size=5,
    ):
        assert ds_length >= static_n_sample if static_n_sample is not None else True
        # operations below need the index to starts from 0
        self.df = df.reset_index(drop=True)
        self.id2label, self.label2id = self.init_id2label_label2id()
        self.ds_length = ds_length
        self.static_n_sample = static_n_sample
        self.frag_height = frag_height
        self.frag_width = frag_width
        self.col_name_prob = col_name_prob
        self.col_name_variant_prob = col_name_variant_prob
        self.schema_process_type = schema_process_type
        self.table_name = table_name

        # These parameters are used to test different fragment types.
        self.max_cols_per_subdf = 1
        self.min_cols_per_subdf = 1
        self.inner_frag = 1
        self.outer_frag = 1

        self.bin_df = pd.DataFrame()
        self.real_cols_bukt_rec = None
        self.real_cols_prob_rec = None
        self.numerical_col_bins = numerical_col_bins
        if numerical_col_bins > 0:
            (
                self.real_cols_bukt_rec,
                self.real_cols_prob_rec,
            ) = self.deal_with_numerical_column()
        self.numerical_col_window_size = numerical_col_window_size

        if self.static_n_sample is not None:
            # store the static dataset here
            self.static_ds = []
            self.generate_static_dataset()

    def deal_with_numerical_column(self):
        real_cols_bukt_rec = {}
        real_cols_prob_rec = {}
        for col in self.df.columns:
            if str(self.df[col].dtype).startswith("int") or str(
                self.df[col].dtype
            ).startswith("float"):
                # All become nan when applying the zscore on a column whose values
                # are  the same, which will cause an error when applying pd.cut
                # if self.df[col].nunique() != 1:
                #     self.df[col] = zscore(self.df[col])
                self.bin_df[col] = self.df[col]
                self.bin_df[col + "bin"] = pd.cut(
                    self.df[col],
                    bins=self.numerical_col_bins,
                    labels=list(range(self.numerical_col_bins)),
                )
                value_cnt = self.bin_df[col + "bin"].value_counts()
                value_cnt /= value_cnt.sum()
                value_cnt = dict(sorted(value_cnt.to_dict().items()))
                real_cols_bukt_rec[col] = list(value_cnt.keys())
                real_cols_prob_rec[col] = list(value_cnt.values())
        return real_cols_bukt_rec, real_cols_prob_rec

    def init_id2label_label2id(self):
        label2id = {}
        id2label = {}
        for cname in self.df.columns:
            label2id[cname] = label2id[cname] if cname in label2id else len(label2id)
            id2label[label2id[cname]] = cname
        return id2label, label2id

    def __len__(self):
        return self.ds_length

    def get_one_random_item(
        self, assigned_sampled_cols_indices=None, assigned_sampled_rows_indices=None
    ):
        # The following code facilitates testing the effects of different fragment types
        # but may seem lengthy and complex. For now, it just constructs regular pairwise fragments.

        loop_cnt = 0

        outer_frag = self.outer_frag
        inner_frag = self.inner_frag
        if outer_frag == 1:
            inner_frag = 1
            first_frag_rows = (
                random.sample(list(range(len(self.df))), self.frag_height)
                if self.col_name_prob <= 1
                else None
            )
            second_frag_rows = (
                random.sample(list(range(len(self.df))), self.frag_height)
                if self.col_name_prob <= 1
                else None
            )

        # When the sentence built from a fragment is too long for the model,
        # go for a resample. With a reasonable setting, the resample merely happens,
        # but I will leave it here.
        while True:
            origin_cols_in_order = self.df.columns.tolist()
            if assigned_sampled_cols_indices is not None:
                # When the columns of pairwise fragments are externally specified.
                # Mainly to leverage this code for constructing general column samples and
                # reducing redundant code writing.
                if not isinstance(assigned_sampled_cols_indices, list):
                    sampled_cols_indices = [assigned_sampled_cols_indices]
                else:
                    sampled_cols_indices = assigned_sampled_cols_indices
            else:
                # This is the regular training logic, i.e., randomly selecting columns for fragments.
                sampled_cols_indices = random.sample(
                    range(len(origin_cols_in_order)), self.frag_width
                )

            sampled_cols_names = [origin_cols_in_order[i] for i in sampled_cols_indices]

            cols_index_batches = []
            rows_index_batches = []

            # keep a counter
            sample_counts = {c: 2 for c in sampled_cols_names}

            # This does not mean anything to this framework anymore
            maintain_cols_order = False
            # if self.maintain_fragment_cols_order_prob > 0 and random.random() < self.maintain_fragment_cols_order_prob:
            #     maintain_cols_order = True

            # Prepare the sampling position information for fragment columns in pairwise fragments.
            # Although it may be cumbersome, it provides more flexibility, such as comparing
            # the effects of different fragment construction methods.
            while any(count > 0 for count in sample_counts.values()):
                num_cols = random.randint(
                    self.min_cols_per_subdf, self.max_cols_per_subdf
                )
                idx_left = [
                    idx_index
                    for idx_index, idx in zip(sampled_cols_indices, sampled_cols_names)
                    if sample_counts[idx] == 2
                ]
                if len(idx_left) > 0:
                    if len(idx_left) >= num_cols:
                        cols_to_sample = random.sample(idx_left, num_cols)
                    else:
                        cols_to_sample = idx_left
                    if outer_frag == 1:
                        rows_index_batches.append(first_frag_rows)
                    else:
                        rows_index_batches.append(
                            random.sample(list(range(len(self.df))), self.frag_height)
                        )
                else:
                    idx_left = [
                        idx_index
                        for idx_index, idx in zip(
                            sampled_cols_indices, sampled_cols_names
                        )
                        if sample_counts[idx] == 1
                    ]
                    if len(idx_left) >= num_cols:
                        cols_to_sample = random.sample(idx_left, num_cols)
                    else:
                        cols_to_sample = idx_left
                    if outer_frag == 1:
                        rows_index_batches.append(second_frag_rows)
                    else:
                        rows_index_batches.append(
                            random.sample(list(range(len(self.df))), self.frag_height)
                        )

                if maintain_cols_order:
                    cols_idx_to_sample = [
                        origin_cols_in_order[i] for i in sorted(cols_to_sample)
                    ]
                else:
                    cols_idx_to_sample = [
                        origin_cols_in_order[i] for i in cols_to_sample
                    ]

                for index in cols_idx_to_sample:
                    sample_counts[index] -= 1
                cols_index_batches.append(cols_idx_to_sample)

            batch_sent_ls = []
            batch_label_ls = []
            for cols_idx_ls, rows_idx_ls in zip(cols_index_batches, rows_index_batches):
                sent_ls = []
                label_ls = []
                # If in the instance-based mode or hybrid mode, where the values in tables are used
                if self.col_name_prob <= 1:
                    # prepare the data to be used
                    if inner_frag == 1:
                        rows = (
                            assigned_sampled_rows_indices
                            if assigned_sampled_rows_indices is not None
                            else rows_idx_ls
                        )
                        sub_df = self.df.loc[rows, cols_idx_ls]
                    else:
                        sub_df = pd.DataFrame(columns=cols_idx_ls)
                        rows = (
                            assigned_sampled_rows_indices
                            if assigned_sampled_rows_indices is not None
                            else [
                                random.sample(
                                    list(range(len(self.df))), self.frag_height
                                )
                                for _ in range(len(cols_idx_ls))
                            ]
                        )
                        for col, row in zip(cols_idx_ls, rows):
                            values = self.df.loc[row, col].tolist()
                            sub_df[col] = values

                    for cname, cdata in sub_df.items():
                        cdata = self.deal_with_sent(cname, cdata)
                        sent_ls.append(cdata)
                        label_ls.append(self.label2id[cname])
                else:
                    for cname in cols_idx_ls:
                        cdata = self.deal_with_sent(cname, None)
                        sent_ls.append(cdata)
                        label_ls.append(self.label2id[cname])

                assert len(sent_ls) == len(label_ls)
                sent = " ".join(sent_ls)
                batch_sent_ls.append(sent)
                batch_label_ls.append(label_ls)

            #  This is just a test to ensure the model can handle the length of the sentence.
            #  The encoded results will not be used in the model.
            input_ids = self.tokenizer.batch_encode_plus(batch_sent_ls)["input_ids"]
            if all(
                len(id)
                <= TableMultiColRandomIntersectStreamDataset.max_model_input_length
                for id in input_ids
            ):
                break
            else:
                loop_cnt += 1
                if loop_cnt == 5:
                    print(
                        "Size settings may cause the sentence being too long for the model, 5 resamples have been done."
                    )
                elif loop_cnt > 20:
                    sys.exit(
                        "over 20 resamples have been done to build one sample, consider change size settings."
                    )

        text = batch_sent_ls
        label = batch_label_ls
        return {"data": text, "label": label}

    def generate_static_dataset(self):
        # frag_width = self.frag_width
        # cols_indices = list(range(len(self.id2label)))

        # cols_counter_dict = {c: 0 for c in cols_indices}

        for _ in range(self.static_n_sample):
            self.static_ds.append(self.get_one_random_item())
            # if self.equal_cols_proceed_mode:
            #     min_value = min(cols_counter_dict.values())
            #     cols_with_more_quota = [key for key, value in cols_counter_dict.items() if value == min_value]
            #     if len(cols_with_more_quota) >= frag_width:
            #         chosen_cols = random.sample(cols_with_more_quota, frag_width)
            #     else:
            #         cols_indices_temp = [i for i in cols_indices if i not in cols_with_more_quota]
            #         chosen_cols = cols_with_more_quota + random.sample(cols_indices_temp,
            #                                                             frag_width - len(cols_with_more_quota))
            # else:
            #     chosen_cols = random.sample(cols_indices, frag_width)
            #
            # random.shuffle(chosen_cols)
            # self.static_ds.append(self.get_one_random_item(chosen_cols))
            #
            # for index in chosen_cols:
            #     cols_counter_dict[index] += 1

    def __getitem__(self, idx):
        # random dataset
        if self.static_n_sample is None:
            return self.get_one_random_item()
        else:
            # static dataset
            cyc_index = idx % len(self.static_ds)
            if cyc_index == 0:
                random.shuffle(self.static_ds)
            return self.static_ds[cyc_index]

    def deal_with_sent(self, cname, cdata):
        if cdata is not None:
            cdata = cdata.tolist()
        # For special treatment of the numerical column, which will contain extended numbers
        # like ['1 2 2 4 2','4 6 2 4 6','5 3 5 7 8'], if filled
        extended_num_cdata = []
        # All cell instance replaced with the column name
        if self.col_name_prob > 1:
            cdata = [self.schema_name_transform(cname) for _ in range(self.frag_height)]
        else:
            if self.numerical_col_bins > 0:
                if cname in self.real_cols_bukt_rec:
                    for n in cdata:
                        converted_dist_repr = self.extend_numerical_element(n, cname)
                        extended_num_cdata.append(converted_dist_repr)
                    # cdata = extended_temp_cdata

        if len(extended_num_cdata) > 0:
            # if self.preserve_orig_numerical_val == 0:
            # extended_num_cdata = ' \ '.join(extended_num_cdata)
            # cdata = ' '.join(extended_num_cdata)
            #     cdata = extended_num_cdata
            # elif self.preserve_orig_numerical_val == 1:
            #     cdata = [str(ent) for ent in cdata]
            #     cdata = [cdata[i] + extended_num_cdata[i] for i in range(len(cdata))]
            #     # cdata = ' '.join(cdata)
            # else:
            #     print('Wrong parameter setting')
            #     sys.exit()

            cdata = [str(ent) for ent in cdata]
            cdata = [cdata[i] + extended_num_cdata[i] for i in range(len(cdata))]

            # cdata = [str(ent) for ent in cdata]
            # cdata = ' '.join(cdata)
            # cdata = cdata + ' [SEP] ' + extended_num_cdata
        else:
            cdata = [str(ent) for ent in cdata]
            # cdata = ' '.join(cdata)
            # if str(self.df[cname].dtype).startswith('int') or str(self.df[cname].dtype).startswith('float'):
            #     # cdata = ' [SEP] '.join(cdata)
            #     # cdata = ' [PAD] '.join(cdata)
            #     # cdata = ' '.join(cdata)
            # else:
            #     cdata = ' | '.join(cdata)
            #     # cdata = '('+') ('.join(cdata)+')'

        # if str(self.df[cname].dtype).startswith('int') or str(self.df[cname].dtype).startswith('float'):
        #     type_special_sign = ' | [NUM] | '
        # else:
        #     type_special_sign = ' | [STR] | '
        # cdata = '[CLS] ' + type_special_sign + cdata + ' [SEP]'

        # Insert the column name with a certain probability
        if self.col_name_prob > 0:
            tp_cdata = []
            for cd in cdata:
                if random.random() < self.col_name_prob:
                    tp_cname = self.schema_name_transform(cname)
                    tp_cdata.append(tp_cname)
                else:
                    tp_cdata.append(cd)
            cdata = tp_cdata
        cdata = " ".join(cdata)
        cdata = "[CLS] " + cdata + " [SEP]"
        return cdata

    def extend_numerical_element(self, n, cname):
        # span_start = max(0, n - (self.numerical_col_window_size - 1) // 2)
        # span_end = min(self.numerical_bins - 1, n + (self.numerical_col_window_size - 1) // 2)
        # # the k here does not have to be the same as the window size, but I will leave it be for now
        # converted_ls = random.choices(self.real_cols_bukt_rec[cname][span_start:span_end + 1],
        #                               self.real_cols_prob_rec[cname][span_start:span_end + 1],
        #                               k=self.numerical_col_window_size)
        # converted_dist_repr = str(n) + ' - ' + ' '.join([str(i) for i in converted_ls])

        start_num = self.bin_df.loc[self.bin_df[cname] == n, cname + "bin"].values[0]
        num_list = [start_num]
        for i in range(self.numerical_col_window_size - 1):
            curr_num = num_list[-1]
            prob_vector = np.zeros(len(self.real_cols_prob_rec[cname]))
            if curr_num > 0:
                prob_vector[curr_num - 1] = self.real_cols_prob_rec[cname][curr_num - 1]
            if curr_num < len(self.real_cols_prob_rec[cname]) - 1:
                prob_vector[curr_num + 1] = self.real_cols_prob_rec[cname][curr_num + 1]
            # There are times when the nearby bin at the left or right side of the current number contains no elements
            if np.sum(prob_vector) == 0:
                break
            prob_vector /= np.sum(prob_vector)
            next_num = np.random.choice(self.real_cols_bukt_rec[cname], p=prob_vector)
            num_list.append(next_num)

        converted_dist_repr = " ".join([str(i) for i in num_list])
        return converted_dist_repr

    def col_name_aug_with_table_name(self, col_name, table_name):
        if len(col_name.split(" ")) > 1:
            col_name = "".join(
                [
                    w[0].upper() + w[1:] if len(w) > 1 else w[0].upper()
                    for w in col_name.split(" ")
                ]
            )
        return table_name + "_" + col_name

    def col_name_abbreviate(self, col_name):
        abbreviation = []
        col_name = "".join([w[0].upper() + w[1:] for w in col_name.split(" ")])
        words = col_name.split("_")
        if len(words) != 1:
            for w in words:
                if len(w) >= 1:
                    bound = random.randint(math.ceil(len(w) / 4), math.ceil(len(w) / 2))
                else:
                    bound = 1
                abbreviation += w[:bound]
            abbreviation = "".join(abbreviation)
            abbreviation = abbreviation.upper()
        else:
            capitals = re.findall("^[a-z]+|[A-Z][a-z]*|\d+", col_name)
            if len(capitals) >= 1:
                for c in capitals:
                    if len(c) > 1:
                        bound = random.randint(
                            math.ceil(len(c) / 4), math.ceil(len(c) / 2)
                        )
                    else:
                        bound = 1
                    abbreviation += [c[:bound]]
                abbreviation = "".join(abbreviation)
        return abbreviation

    def col_name_drop_vowels(self, col_name):
        word = "".join(
            [
                w[0].upper() + w[1:] if len(w) > 1 else w[0].upper()
                for w in col_name.split(" ")
            ]
        )
        table = str.maketrans(dict.fromkeys("aeiouyAEIOUY"))
        return word.translate(table).lower()

    def schema_name_transform(self, col_name):
        name_type = self.schema_process_type // 10
        char_type = self.schema_process_type % 10

        type = random.choices(
            [0, 1],
            weights=[self.col_name_variant_prob, 1 - self.col_name_variant_prob],
            k=1,
        )[0]
        if type == 0:
            if name_type == 0:
                # original mame
                return col_name
            elif name_type == 1:
                # Construct fabricated schema names in the manner of Valentine.
                trans_type = random.randint(1, 5)
                if trans_type == 1:
                    trans_col_name = self.col_name_aug_with_table_name(
                        col_name, self.table_name
                    )
                elif trans_type == 2:
                    trans_col_name = self.col_name_abbreviate(col_name)
                elif trans_type == 3:
                    trans_col_name = self.col_name_drop_vowels(col_name)
                elif trans_type == 4:
                    trans_col_name = self.col_name_abbreviate(col_name)
                    trans_col_name = self.col_name_aug_with_table_name(
                        trans_col_name, self.table_name
                    )
                elif trans_type == 5:
                    trans_col_name = self.col_name_drop_vowels(col_name)
                    trans_col_name = self.col_name_aug_with_table_name(
                        trans_col_name, self.table_name
                    )
                return trans_col_name
            # a wrong idea
            # elif name_type == 2:
            #     # drop chars from the column names
            #     tp_name = ''
            #     for c in col_name:
            #         if random.random() > 0.15:
            #             tp_name += c
            #     return tp_name
        elif type == 1:
            # char wise with random dropout
            name_char_ls = [c for c in col_name]
            if char_type == 0:
                pass
            # can boost the performance a little
            elif char_type == 1:
                if len(name_char_ls) > 5:
                    name_char_ls = [x for x in name_char_ls if random.random() > 0.15]
            tp_name = " ".join(name_char_ls)
            return tp_name

    def perturb_data(self, cname, np_cdata):
        df_mean = self.df[cname].mean()
        df_stdev = self.df[cname].std()

        sign = random.sample([-1, 1], 1)[0]
        perturb = sign * random.randint(10, 50) / 100

        p_mean = df_mean + df_mean * perturb
        p_stdev = df_stdev + df_stdev * perturb

        new_cdata = np.random.normal(p_mean, p_stdev, np_cdata.size)

        return new_cdata.tolist()

    @staticmethod
    def read_and_process_table(
        src_csv_path,
        tgt_csv_path,
        golden_match_file,
        dropna,
        col_name_prob,
    ):
        def get_orig_df(csv_path, col_name_prob):
            # Check if the file has two header rows since this will happen in some csv files
            # in the dataset, which will cause some trouble when operations are related to
            # the datatype of a column.
            csv_read_skiprows = 0
            with open(csv_path, "r") as f:
                line1 = f.readline()
                line2 = f.readline()
                if line1 == line2:
                    csv_read_skiprows = 1
            if col_name_prob <= 1:
                orig_df = pd.read_csv(
                    csv_path, index_col=False, skiprows=csv_read_skiprows
                )
            else:
                # schema-based, no need for the instances
                orig_df = pd.read_csv(
                    csv_path, index_col=False, skiprows=csv_read_skiprows, nrows=0
                )
            return orig_df

        def deal_with_empty(orig_df, empty_cols_ls, dropna, col_name_prob):
            # better not to simply drop the nan since there exist columns whose value is most nan,
            # drop will leave the whole dataset to just a few rows
            if dropna:
                orig_df.dropna(inplace=True)
            # when >1, it's in schema-based mode, no need to deal with empty cols
            elif col_name_prob <= 1:
                # empty_cols = orig_df.columns[orig_df.replace('', np.nan).count() < 0.2 * len(orig_df)]
                empty_cols = orig_df.columns[(orig_df.isna() | (orig_df == "")).all()]
                empty_cols_ls.extend(list(empty_cols))

        if isinstance(src_csv_path, pd.DataFrame):
            src_orig_df = src_csv_path
        else:
            src_orig_df = get_orig_df(src_csv_path, col_name_prob)
        if isinstance(tgt_csv_path, pd.DataFrame):
            tgt_orig_df = tgt_csv_path
        else:
            tgt_orig_df = get_orig_df(tgt_csv_path, col_name_prob)
        src_empty_cols = []
        tgt_empty_cols = []
        deal_with_empty(src_orig_df, src_empty_cols, dropna, col_name_prob)
        deal_with_empty(tgt_orig_df, tgt_empty_cols, dropna, col_name_prob)

        src_table_name = "source"
        tgt_table_name = "target"

        # NO such cases in the dataset, no use, invalid code
        # golden_matches = GoldenStandardLoader(golden_match_file)
        # src_to_tgt = {match[0]: match[1] for match in golden_matches.expected_matches}
        # tgt_to_src = {match[1]: match[0] for match in golden_matches.expected_matches}
        # for empty_col in src_empty_cols:
        #     if src_to_tgt.get((src_table_name, empty_col)) is not None:
        #         tgt_empty_cols.append(src_to_tgt.get((src_table_name, empty_col))[1])
        # for empty_col in tgt_empty_cols:
        #     if tgt_to_src.get((tgt_table_name, empty_col)) is not None:
        #         src_empty_cols.append(tgt_to_src.get((tgt_table_name, empty_col))[1])

        src_orig_df = src_orig_df.drop(columns=src_empty_cols)
        tgt_orig_df = tgt_orig_df.drop(columns=tgt_empty_cols)

        golden_standard = None
        num_golden_standard_set = set()
        if golden_match_file is not None:
            golden_standard = GoldenStandardLoader(
                golden_match_file, src_empty_cols, tgt_empty_cols
            )

            num_golden_standard_set = set()
            src_num_cols_ls = []
            for col in src_orig_df.columns:
                if str(src_orig_df[col].dtype).startswith("int") or str(
                    src_orig_df[col].dtype
                ).startswith("float"):
                    src_num_cols_ls.append(col)
            for mt in golden_standard.expected_matches:
                tp_matche_ls = list(mt)
                src_cname = (
                    tp_matche_ls[0][1]
                    if "source" in tp_matche_ls[0][0]
                    else tp_matche_ls[1][1]
                )
                if src_cname in src_num_cols_ls:
                    num_golden_standard_set.add(mt)

        ### NOTE: the following does not mean anything to this framework anymore

        # default max_distance=2 makes that the columns stay near after the permutation,
        # consistent with the real world situation
        # def special_permutation(lst, max_distance=2):
        #     n = len(lst)
        #     for i in range(n - 1):
        #         start = max(i - max_distance, 0)
        #         end = min(i + max_distance, n - 1)
        #         j = random.randint(start, end)
        #         lst[i], lst[j] = lst[j], lst[i]
        #     return lst

        # # arrange the position of the matching columns to test the effect of the column order
        # src_match_cols = []
        # tgt_match_cols = []
        # def merge_list_randomly(origin_cols_ls, match_cols_ls, maintain_matches_order=True):
        #     target_cols_ls = [None] * len(origin_cols_ls)
        #     matches_insert_pos = random.sample(range(len(origin_cols_ls)), len(match_cols_ls))
        #     if maintain_matches_order:
        #         matches_insert_pos.sort()
        #     for i, element in enumerate(match_cols_ls):
        #         target_cols_ls[matches_insert_pos[i]] = element
        #     none_pos = [i for i, v in enumerate(target_cols_ls) if v is None]
        #     rest_cols_to_insert = [col for col in origin_cols_ls if col not in match_cols_ls]
        #     assert len(none_pos) == len(rest_cols_to_insert)
        #     for p, col in zip(none_pos, rest_cols_to_insert):
        #         target_cols_ls[p] = col
        #     return target_cols_ls
        #
        # for match in golden_standard.expected_matches:
        #     match1, match2 = match
        #     if 'source' in match1[0]:
        #         src_match_cols.append(match1[1])
        #         tgt_match_cols.append(match2[1])
        #     else:
        #         src_match_cols.append(match2[1])
        #         tgt_match_cols.append(match1[1])
        #
        # paired_match_cols = list(zip(src_match_cols, tgt_match_cols))
        # paired_match_cols = sorted(paired_match_cols, key=lambda x: x[0])
        # src_match_cols, tgt_match_cols = map(list, zip(*paired_match_cols))
        #
        # if permut_cols_order == 0:
        #     src_target_column_list = src_match_cols + [col for col in src_orig_df.columns.tolist() if
        #                                                col not in src_match_cols]
        #     tgt_target_column_list = tgt_match_cols + [col for col in tgt_orig_df.columns.tolist() if
        #                                                col not in tgt_match_cols]
        # elif permut_cols_order == 1:
        #     random.shuffle(src_match_cols)
        #     random.shuffle(tgt_match_cols)
        #     src_target_column_list = src_match_cols + [col for col in src_orig_df.columns.tolist() if
        #                                                col not in src_match_cols]
        #     tgt_target_column_list = tgt_match_cols + [col for col in tgt_orig_df.columns.tolist() if
        #                                                col not in tgt_match_cols]
        # elif permut_cols_order == 2:
        #     src_target_column_list = merge_list_randomly(src_orig_df.columns.tolist(), src_match_cols, True)
        #     tgt_target_column_list = merge_list_randomly(tgt_orig_df.columns.tolist(), tgt_match_cols, True)
        # elif permut_cols_order == 3:
        #     src_target_column_list = merge_list_randomly(src_orig_df.columns.tolist(), src_match_cols, False)
        #     tgt_target_column_list = merge_list_randomly(tgt_orig_df.columns.tolist(), tgt_match_cols, False)
        # else:
        #     sys.exit()
        #
        # src_orig_df = src_orig_df[src_target_column_list]
        # tgt_orig_df = tgt_orig_df[tgt_target_column_list]

        return (
            (src_table_name, src_orig_df),
            (tgt_table_name, tgt_orig_df),
            golden_standard,
            num_golden_standard_set,
        )

    @staticmethod
    def deal_with_nan_values(df):
        # # fill NaN values in numerical columns with the mean value of that column
        # num_cols = df.select_dtypes(include=[np.number]).columns
        # for col in num_cols:
        #     # if the decimal part of the number is all zero
        #     def is_decimal_zero(x):
        #         if pd.isna(x):
        #             return True
        #         else:
        #             return (x - int(x)) == 0
        #     # there is no all-nan columns as I removed them
        #     if df[col].apply(is_decimal_zero).all():
        #         # if the values in the columns is just integer and nan, I may should not inject many float numbers
        #         mean_value = math.floor(df[col].mean())
        #     else:
        #         mean_value = df[col].mean()
        #     df[col].fillna(mean_value, inplace=True)
        #
        # # fill NaN values in non-numerical columns with the most existing value of that column
        # non_num_cols = df.select_dtypes(exclude=[np.number]).columns
        # for col in non_num_cols:
        #     df[col].replace('', np.nan, inplace=True)
        #     # sometimes all the value is nan in the original dataset, these columns will be removed in the
        #     # instance-based matching mode
        #     most_existing_value = df[col].mode()[0]
        #     df[col].fillna(most_existing_value, inplace=True)

        # for col in df.columns:
        #     df[col].replace('', np.nan, inplace=True)
        #     non_nan_values = df[col].dropna().unique()
        #     # non_nan_values = df[col].unique()
        #     df[col] = df[col].apply(lambda x: np.random.choice(non_nan_values) if pd.isna(x) else x)

        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            non_nan_values = df[col].dropna()
            df[col] = df[col].apply(
                lambda x: np.random.choice(non_nan_values) if pd.isna(x) else x
            )

        non_num_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_num_cols:
            df[col].replace("", np.nan, inplace=True)
            non_nan_values = df[col].dropna().unique()
            df[col] = df[col].apply(
                lambda x: np.random.choice(non_nan_values) if pd.isna(x) else x
            )

    @classmethod
    def get_dataset(
        cls,
        df,
        table_name,
        frag_height,
        frag_width,
        ds_length,
        static_n_sample=None,
        n_sub_rows_portion=0,
        col_name_prob=0.0,
        col_name_variant_prob=0.5,
        schema_process_type=None,
        numerical_col_bins=0,
        numerical_col_window_size=5,
        model_loc=None,
    ):
        # cls.tokenizer = BertTokenizer.from_pretrained(
        #     'bert-base-uncased' if model_loc is None else os.path.join(model_loc, 'bert-base-uncased'))
        cls.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        cls.max_model_input_length = 512

        orig_df = df

        # when > 1, it's in schema-based mode
        if col_name_prob <= 1:
            cls.deal_with_nan_values(orig_df)

        # use only part of the rows of the original table
        if n_sub_rows_portion < 1:
            n_sub_rows = round(len(orig_df) * n_sub_rows_portion)
            # no less than the length of one sample
            n_sub_rows = max(frag_height, n_sub_rows)
            orig_df = orig_df.sample(n=n_sub_rows)

        # num_cols = orig_df.select_dtypes(include=[np.number]).columns
        # orig_df = orig_df.drop(columns=[col for col in df.columns if col not in num_cols])

        # num_cols = df.select_dtypes(include=[np.number]).columns
        # for col in num_cols:
        #     non_nan_values = df[col].dropna()
        #     df[col] = df[col].apply(lambda x: np.random.choice(non_nan_values) if pd.isna(x) else x)
        #
        # non_num_cols = df.select_dtypes(exclude=[np.number]).columns
        # for col in non_num_cols:
        #     df[col].replace('', np.nan, inplace=True)
        #     non_nan_values = df[col].dropna().unique()
        #     df[col] = df[col].apply(lambda x: np.random.choice(non_nan_values) if pd.isna(x) else x)

        orig_df = orig_df.sample(frac=1)
        trn_ds = cls(
            df=orig_df,
            ds_length=ds_length,
            frag_height=frag_height,
            frag_width=frag_width,
            table_name=table_name,
            static_n_sample=static_n_sample,
            col_name_prob=col_name_prob,
            col_name_variant_prob=col_name_variant_prob,
            schema_process_type=schema_process_type,
            numerical_col_bins=numerical_col_bins,
            numerical_col_window_size=numerical_col_window_size,
        )
        return trn_ds

    @staticmethod
    def collate_fn(batches):
        batch_text_ls = [sent for b in batches for sent in b["data"]]

        data = TableMultiColRandomIntersectStreamDataset.tokenizer(
            batch_text_ls,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=TableMultiColRandomIntersectStreamDataset.max_model_input_length,
        )
        labels = []
        for sample in batches:
            labels.append(sample["label"])
        batch = {"data": data, "label": labels}
        return batch


class FixedSubDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: TableMultiColRandomIntersectStreamDataset, n_cols=None):
        assert n_cols % 2 == 0

        self.id2label = dataset.id2label
        self.label2id = dataset.label2id

        self.sub_ds = []

        frag_height = dataset.frag_height
        cols_indices = list(range(len(dataset.id2label)))

        # two datapoints in each pairwise fragments, so divide by 2
        cols_counter_dict = {c: n_cols / 2 for c in cols_indices}

        # The following somewhat combersome code ensures that the occurrences of
        # each column in randomly generated fragments are the same (to ensure
        # that each fragment has the same size, some columns may appear an extra time).
        while any(count > 0 for count in cols_counter_dict.values()):
            rest_cols_counter = {
                key: value for key, value in cols_counter_dict.items() if value > 0
            }
            if len(rest_cols_counter) >= frag_height:
                max_value = max(rest_cols_counter.values())
                cols_with_more_quota = [
                    key
                    for key, value in rest_cols_counter.items()
                    if value == max_value
                ]

                if len(cols_with_more_quota) >= frag_height:
                    chosen_cols_ = random.sample(cols_with_more_quota, frag_height)
                else:
                    cols_indices_temp = [
                        i for i in cols_indices if i not in cols_with_more_quota
                    ]
                    chosen_cols_ = cols_with_more_quota + random.sample(
                        cols_indices_temp, frag_height - len(cols_with_more_quota)
                    )
                chosen_cols = chosen_cols_
            else:
                chosen_cols_ = [key for key, value in rest_cols_counter.items()]
                cols_indices_temp = [i for i in cols_indices if i not in chosen_cols_]
                chosen_cols = chosen_cols_ + random.sample(
                    cols_indices_temp, frag_height - len(chosen_cols_)
                )

            random.shuffle(chosen_cols)
            self.sub_ds.append(dataset.get_one_random_item(chosen_cols))
            for index in chosen_cols_:
                cols_counter_dict[index] -= 1

    def __len__(self):
        return len(self.sub_ds)

    def __getitem__(self, idx):
        return self.sub_ds[idx]


# This is just too too too slow
# class SoloColsDataset(torch.utils.data.Dataset):
#
#     def __init__(self, dataset: TableMultiColRandomIntersectStreamDataset, n_cols=20):
#         # assert n_cols % 2 == 0
#
#         self.sub_ds = []
#         for col_idx in range(len(dataset.id2label)):
#             # for i in range(n_cols//2):
#             for i in range(n_cols):
#                 # item_dc = {'data': text, 'label': label}
#                 item_dc = dataset.get_one_random_item(col_idx)
#                 # so that one-shot for each column is possible
#                 item_dc = {'data': [item_dc['data'][0]], 'label': [item_dc['label'][0]]}
#                 self.sub_ds.append(item_dc)
#
#     def __len__(self):
#         return len(self.sub_ds)
#
#     def __getitem__(self, idx):
#         return self.sub_ds[idx]


class SoloColsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: TableMultiColRandomIntersectStreamDataset, n_cols=200):
        self.dataset = dataset
        self.cols_ls = [col_idx for col_idx in range(len(dataset.id2label))] * n_cols
        if len(self.dataset.df) == 0:
            # in schema-based mode
            self.rows_ls = None
        else:
            self.rows_ls = [
                random.sample(
                    list(range(len(self.dataset.df))), self.dataset.frag_height
                )
                for _ in range(len(self.cols_ls))
            ]
            random.shuffle(self.rows_ls)
        # not necessary
        random.shuffle(self.cols_ls)

    def __len__(self):
        return len(self.cols_ls)

    def __getitem__(self, idx):
        # borrow the get_one_random_item to save some typing, but the content
        # returned will be double, simply take the first one
        item_dc = self.dataset.get_one_random_item(
            self.cols_ls[idx], self.rows_ls[idx] if self.rows_ls is not None else None
        )
        return {"data": [item_dc["data"][0]], "label": [item_dc["label"][0]]}
