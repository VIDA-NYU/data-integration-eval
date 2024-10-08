import argparse
import json
import math
import os
import sys
import time
import warnings

import src.metrics.metrics as module_metric
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_samples, silhouette_score
from src.data_loader.sm_dataset import (
    SoloColsDataset,
    TableMultiColRandomIntersectStreamDataset,
)
from src.models.sm_model import BertFoMatching
from src.utils import logconf, utils
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# ignore annoying warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class TrainApp:
    def __init__(
        self,
        n_trn_cols=0,
        batch_size=1,
        frag_height=1,
        frag_width=12,
        learning_rate=3e-5,
        orig_file_src=None,
        orig_file_tgt=None,
        orig_file_golden_matches=None,
        dataset_name=None,  # The name of the dataset.
        col_name_prob=0,  # The probability of using column name information.
        comment=None,  # The comment for the current run.
        store_matches=0,  # 0: not store; 1: store
        ds_scale_factor=1,  # The scale factor of the dataset size.
        process_mode=0,  # 0: Train to match; 1: Train first, then use val data for matching (embeddings centroids approach); 2: No training, directly use val data for matching (embeddings centroids approach).
        check_frequency=0,  # The interval between two performance checking (measured in terms of column frequency).
        n_val_cols=0,  # The frequency of sampling for each column in the val set.
        prop_sub_rows=1,  # What proportion of rows from the original table will be used.
        dropna=False,  # Whether or not to drop na value in the original tabular data.
        temperature=1,  # temperature used in the loss function.
        t_dist_scale_factor=10,  # the scale factor use on the student t-distribution.
        cols_repr_init_std=0.0005,  # The standard deviation of the normal distribution used to initialize column representations.
        output_repr_dim=512,  # The dimensions of the model's output and column representations.
        meta_match_loss_weight=1,  # weight of the meta matching loss.
        agent_delegate_loss_weight=1,  # weight of the agent delegating loss.
        rec_loss_weight=0.2,  # weight of the matching rectification loss.
        sk_n_iter=5,  # The number of iterations in the Sinkhorn-Knopp algorithm.
        sk_reg_weight=20,  # The reciprocal of the weight of entropic regularization in the Sinkhorn-Knopp algorithm.
        schema_process_type=00,  # The type of operation performed on schema names when using them. The first digit represents the name, and the second digit represents the name's characters. 00: Use the original name and characters. 01: Use the original name and characters with random drop (can boost performance a bit). 10: Transform the original name in the same way as during the construction of fabricated data, and characters.
        col_name_variant_prob=0.5,  # The probability of using column name information based on name transformation, including the original name or a transformation based on the name, when in schema-based mode or hybrid mode.
        frag_width_auto_scale_reference=None,  # When this parameter is set, the --frag-width parameter will automatically scale based on this parameter and the table width. This is mainly for testing the impact of different fragment widths on the results. For example, when --n-cols-per-sample is 8, and the current smaller table width is 6, and this parameter is set to 12, then the --n-cols-per-sample will be reset to 4 (6*(8/12)), instead of being set to 6 (the samller table width) in normal case.
        numerical_cols_bins=20,  # The number of bins when bucketing the numerical columns. 0 means that there will be no bucketing, and only the original numerical value will be used.
        numerical_cols_window_size=10,  # the size of the distribution-aware fingerprint strings.
        data_dir="data",  # Location for storing data and outputs, default is project_loc/data.
        out_dir="output",  # Location for storing outputs, default is in project_loc/data/output.
    ):
        self.algo_begin_time = time.time()

        # parser.add_argument('--maintain-fragment-cols-order-prob',
        #                     help='The probability of maintaining the order of columns in fragments'
        #                          ' consistent with the original table.',
        #                     type=float,
        #                     default=0
        #                     )

        # parser.add_argument('--permut-cols-order',
        #                     help='0: All matching columns are gathered together in the same order in both tables; '
        #                          '1: All matching columns are gathered together in a different order in both tables; '
        #                          '2: All matching columns are scattered in the same order in both tables; '
        #                          '3: All matching columns are scattered in a different order in both tables.',
        #                     type=int,
        #                     default=0
        #                     )

        self.ds_scale_factor = ds_scale_factor
        self.process_mode = process_mode
        self.batch_size = batch_size
        self.lr = learning_rate
        self.dataset_name = dataset_name
        self.dropna = dropna
        self.temperature = temperature
        self.golden_matches_file = orig_file_golden_matches
        self.col_name_prob = col_name_prob
        self.output_repr_dim = output_repr_dim
        self.comment = comment
        self.check_freq = check_frequency
        self.col_name_variant_prob = col_name_variant_prob
        self.schema_process_type = schema_process_type
        self.orig_file_src = orig_file_src
        self.orig_file_tgt = orig_file_tgt
        self.frag_height = frag_height
        self.frag_width = frag_width
        self.frag_width_auto_scale_reference = frag_width_auto_scale_reference
        self.t_dist_scale_factor = t_dist_scale_factor
        self.numerical_col_bins = numerical_cols_bins
        self.numerical_col_window_size = numerical_cols_window_size
        self.meta_match_loss_weight = meta_match_loss_weight
        self.agent_delegate_loss_weight = agent_delegate_loss_weight
        self.rec_loss_weight = rec_loss_weight
        self.sk_n_iter = sk_n_iter
        self.sk_reg_weight = sk_reg_weight
        self.cols_repr_init_std = cols_repr_init_std
        self.n_trn_cols = n_trn_cols
        self.n_val_cols = n_val_cols
        self.data_dir = data_dir
        self.prop_sub_rows = prop_sub_rows
        self.out_dir = out_dir
        self.store_matches = store_matches

        if self.process_mode in [0]:
            assert self.n_trn_cols > 0, "specific --n-trn-cols > 0 needed"
            self.n_val_cols = 0
        elif self.process_mode in [1, 2]:
            # any --n-trn-cols > 0 needed, (only for building the val data)
            self.n_trn_cols = 10

        ############################################
        # todo: delete this
        # run_id = '-'.join([d for d in self.dataset_name.split('/')])
        # for _,_,files in os.walk(os.path.join(self.data_dir,'output',self.out_dir,self.comment)):
        #     full_rec=[f[20:] for f in files]
        # print('*************************************************')
        # print(os.path.join(self.data_dir,'output',self.out_dir,self.comment))
        # if run_id+'.json' in full_rec:
        #     print('no need to run, exit.')
        #     sys.exit()
        ############################################

        # if self.dataset_name is None:
        #     self.dataset_name='/'.join(self.orig_file_src.split('/')[-3:-1])

        (
            (self.src_table_name, self.src_df),
            (self.tgt_table_name, self.tgt_df),
            self.golden_matches,
            self.numerical_golden_matches_set,
        ) = TableMultiColRandomIntersectStreamDataset.read_and_process_table(
            self.orig_file_src,
            self.orig_file_tgt,
            self.golden_matches_file,
            self.dropna,
            self.col_name_prob,
        )

        self.align_fragment_size_setting()

        self.logger = self.init_logger()

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.logger.info("Using GPU.")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU.")

        ####################################################################
        self.logger.info("begin to prepare the data ...")

        (
            self.src_trn_ds,
            self.src_val_ds,
            self.tgt_trn_ds,
            self.tgt_val_ds,
        ) = self.init_dataset()

        # when to perform checks
        ds_length = len(self.src_trn_ds)
        if self.check_freq == 0:
            self.check_point_cols_ls = [self.n_trn_cols]
            self.check_point_sample_ls = [ds_length]
        else:
            # The total train column corresponds to the full length ds_length that has already been scaled
            # based on the dataset scale factor ds_scale_factor
            total_trn_col = int(self.n_trn_cols * self.ds_scale_factor)
            self.check_point_cols_ls = [
                i for i in range(self.check_freq, total_trn_col, self.check_freq)
            ]
            self.check_point_sample_ls = [
                math.ceil((ds_length / total_trn_col) * cp)
                for cp in self.check_point_cols_ls
            ]
            self.check_point_cols_ls.append(total_trn_col)
            self.check_point_sample_ls.append(ds_length)

        if self.process_mode != 0:
            # if not in trn-to-match mode, there are all single columns in the val dataset,
            # increase the batch size
            val_batch_size = self.batch_size * self.frag_width * 2
        else:
            val_batch_size = self.batch_size

        self.src_trn_dl = self.init_dl(self.src_trn_ds, self.batch_size)
        self.src_val_dl = (
            None
            if self.n_val_cols <= 0
            else self.init_dl(self.src_val_ds, val_batch_size)
        )
        self.tgt_trn_dl = self.init_dl(self.tgt_trn_ds, self.batch_size)
        self.tgt_val_dl = (
            None
            if self.n_val_cols <= 0
            else self.init_dl(self.tgt_val_ds, val_batch_size)
        )

        self.logger.info(f"src effective table width: {self.src_df.shape[1]}")
        self.logger.info(f"tgt effective table width: {self.tgt_df.shape[1]}")

        # if self.ds_scale_factor == 1:
        #     self.logger.info('random data samples')
        # else:
        #     self.logger.info('static type of data applied')
        #     self.logger.info(f'src trn data size: {len(self.src_trn_ds)}')
        #     self.logger.info(f'tgt trn data size: {len(self.tgt_trn_ds)}')
        #
        # if self.n_val_cols <= 0:
        #     self.logger.info('No val data')
        # else:
        #     self.logger.info(f'src val data size: {len(self.src_val_ds)}')
        #     self.logger.info(f'tgt val data size: {len(self.tgt_val_ds)}')

        ####################################################################

        # # record the params of the running
        # self.running_params = []
        # for arg in vars(self.args):
        #     self.running_params.append('{}={}'.format(arg, getattr(self.args, arg)))
        #
        # curr_run_file = os.path.splitext(os.path.basename(__file__))[0]
        # self.running_params.append('running_app={}'.format(curr_run_file))
        #
        # self.running_params.append(
        #     'n_src_trn_data_size_={}'.format(len(self.src_trn_ds)))
        # self.running_params.append(
        #     'n_tgt_trn_data_size_={}'.format(len(self.tgt_trn_ds)))
        # self.running_params.append(
        #     'n_src_val_data_size_={}'.format(len(self.src_val_ds) if self.n_val_cols > 0 else 0))
        # self.running_params.append(
        #     'n_tgt_val_data_size_={}'.format(len(self.tgt_val_ds) if self.n_val_cols > 0 else 0))
        #
        # for param in self.running_params:
        #     self.logger.info(param)

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

        self.metric_fns = self.init_metrics()

        self.best_matches = None
        self.best_metrics = None

        self.prev_sim_matrix = None
        self.curr_sim_matrix = None

        self.curr_metrics = None

        self.curr_numerical_matches_set = set()

        self.src_val_silhouette_score = None
        self.tgt_val_silhouette_score = None

        self.curr_sorted_matches = None

        # self.curr_matches_rank = None
        # self.prev_matches_rank = None

        self.src_cols_used_dict = {
            k: 0 for k in list(range(len(self.src_trn_ds.id2label)))
        }
        self.tgt_cols_used_dict = {
            k: 0 for k in list(range(len(self.tgt_trn_ds.id2label)))
        }

        self.res_dic = {}

        self.src_agent_sh_ls = None
        self.src_center_sh_ls = None
        self.tgt_agent_sh_ls = None
        self.tgt_center_sh_ls = None

        self.algo_comp_begin_time = time.time()

    def update_usecase(self, dataset_name, orig_file_golden_matches):
        self.golden_matches_file = orig_file_golden_matches
        self.dataset_name = dataset_name

        (
            (self.src_table_name, self.src_df),
            (self.tgt_table_name, self.tgt_df),
            self.golden_matches,
            self.numerical_golden_matches_set,
        ) = TableMultiColRandomIntersectStreamDataset.read_and_process_table(
            self.orig_file_src,
            self.orig_file_tgt,
            self.golden_matches_file,
            self.dropna,
            self.col_name_prob,
        )

    def align_fragment_size_setting(self):
        src_n_cols = self.src_df.shape[1]
        tgt_n_cols = self.tgt_df.shape[1]

        # frag_width_auto_scale_reference is for experiment use.
        # Always setting it to be the same as frag_width currently, so it has no impact on the framework.
        self.frag_width_auto_scale_reference = self.frag_width
        # only activate auto scale when the table width is less than it
        if (
            self.frag_width_auto_scale_reference > 0
            and self.frag_width_auto_scale_reference > min(src_n_cols, tgt_n_cols)
        ):
            self.frag_width = round(
                self.frag_width
                * (min(src_n_cols, tgt_n_cols) / self.frag_width_auto_scale_reference)
            )

        self.frag_width = min(src_n_cols, tgt_n_cols, self.frag_width)

        # if self.min_cols_per_subdf < 1:
        #     self.min_cols_per_subdf = 1
        # if self.max_cols_per_subdf > self.frag_width or self.max_cols_per_subdf < 1:
        #     self.max_cols_per_subdf = self.frag_width
        # if self.min_cols_per_subdf > self.max_cols_per_subdf:
        #     self.min_cols_per_subdf = self.max_cols_per_subdf

    def get_output_dir(self):
        data_dir = os.path.join(utils.get_project_root(), self.data_dir)
        return os.path.join(os.path.join(data_dir), f"{self.out_dir}/{self.comment}")

    def init_logger(self):
        logger = logconf.get_simple_print_logger(__name__)
        return logger

    def init_optimizer(self):
        # # no need to do it this way
        # optimizer_model = Adam(itertools.chain.from_iterable(
        #     [m.parameters() for m in self.model.children() if not isinstance(m, AgentsLayer)]),
        #     lr=self.lr)
        # optimizer_centroid = Adam(itertools.chain.from_iterable(
        #     [m.parameters() for m in self.model.children() if isinstance(m, AgentsLayer)]),
        #     lr=self.lr)
        # return optimizer_model, optimizer_centroid

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        # optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer

    def init_dataset(self):
        # Take the table with the larger width as the reference
        df_length_refer_wilder = math.ceil(
            (self.n_trn_cols * max(self.src_df.shape[1], self.tgt_df.shape[1]))
            / (self.frag_width * 2)
        )

        if self.ds_scale_factor == 1:
            ds_length = df_length_refer_wilder
            static_n_samples = None
        elif self.ds_scale_factor > 1:
            ds_length = int(df_length_refer_wilder * self.ds_scale_factor)
            static_n_samples = df_length_refer_wilder

        src_trn_ds = TableMultiColRandomIntersectStreamDataset.get_dataset(
            df=self.src_df,
            table_name=self.src_table_name,
            frag_height=self.frag_height,
            frag_width=self.frag_width,
            ds_length=ds_length,
            static_n_sample=static_n_samples,
            n_sub_rows_portion=self.prop_sub_rows,
            col_name_prob=self.col_name_prob,
            col_name_variant_prob=self.col_name_variant_prob
            if any(ds in self.dataset_name for ds in ["Musicians", "DeepMDatasets"])
            else 0.1,
            schema_process_type=self.schema_process_type,
            numerical_col_bins=self.numerical_col_bins,
            numerical_col_window_size=self.numerical_col_window_size,
            model_loc=None
            if not os.path.exists(os.path.join(self.data_dir, "pretrained_model"))
            else os.path.join(self.data_dir, "pretrained_model"),
        )

        tgt_trn_ds = TableMultiColRandomIntersectStreamDataset.get_dataset(
            df=self.tgt_df,
            table_name=self.tgt_table_name,
            frag_height=self.frag_height,
            frag_width=self.frag_width,
            ds_length=ds_length,
            static_n_sample=static_n_samples,
            n_sub_rows_portion=self.prop_sub_rows,
            col_name_prob=self.col_name_prob,
            col_name_variant_prob=self.col_name_variant_prob
            if any(ds in self.dataset_name for ds in ["Musicians", "DeepMDatasets"])
            else 0.1,
            schema_process_type=self.schema_process_type,
            numerical_col_bins=self.numerical_col_bins,
            numerical_col_window_size=self.numerical_col_window_size,
            model_loc=None
            if self.data_dir is None
            else os.path.join(self.data_dir, "pretrained_model"),
        )

        if self.n_val_cols <= 0:
            src_val_ds = None
            tgt_val_ds = None
        else:
            # src_n_samples = math.ceil((self.n_val_cols * self.src_df.shape[1]) / (self.frag_width * 2))
            # tgt_n_samples = math.ceil((self.n_val_cols * self.tgt_df.shape[1]) / (self.frag_width * 2))

            # if self.process_mode == 1 or self.process_mode == 3:
            #     src_val_ds = torch.utils.data.Subset(src_trn_ds,
            #                                          random.sample(list(range(len(src_trn_ds))), src_n_samples))
            #     tgt_val_ds = torch.utils.data.Subset(tgt_trn_ds,
            #                                          random.sample(list(range(len(tgt_trn_ds))), tgt_n_samples))
            # elif self.process_mode == 2:
            #     src_val_ds = SoloColsDataset(src_trn_ds, self.n_val_cols)
            #     tgt_val_ds = SoloColsDataset(tgt_trn_ds, self.n_val_cols)

            src_val_ds = SoloColsDataset(src_trn_ds, self.n_val_cols)
            tgt_val_ds = SoloColsDataset(tgt_trn_ds, self.n_val_cols)

        return src_trn_ds, src_val_ds, tgt_trn_ds, tgt_val_ds

    def init_dl(self, dataset, bs):
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=TableMultiColRandomIntersectStreamDataset.collate_fn,
        )
        return dataloader

    def init_model(self):
        src_n_cls = len(self.src_trn_ds.id2label)
        tgt_n_cls = len(self.tgt_trn_ds.id2label)
        model = BertFoMatching(
            src_n_cls,
            tgt_n_cls,
            self.output_repr_dim,
            self.t_dist_scale_factor,
            self.temperature,
            self.sk_n_iter,
            self.sk_reg_weight,
            self.cols_repr_init_std,
            os.path.join(self.data_dir, "pretrained_model"),
        )

        return model.to(self.device)

    #  a totally wrong choice
    # def init_col_repr(self):
    #     self.logger.info('initialize the columns representation...')
    #
    #     # use the columns of the randomly built fragments to init
    #     # src_dl = self.src_trn_sub_dl
    #     # tgt_dl = self.tgt_trn_sub_dl
    #
    #     # use the single columns randomly built to init
    #     src_dl = self.src_trn_init_sub_dl
    #     tgt_dl = self.tgt_trn_init_sub_dl
    #
    #     src_final_repr, src_flat_label_ls, _ = self.eval_dataset(src_dl, 'src', return_orig_logits=False)
    #     tgt_final_repr, tgt_flat_label_ls, _ = self.eval_dataset(tgt_dl, 'tgt', return_orig_logits=False)
    #
    #     src_final_repr = np.array(src_final_repr)
    #     src_flat_label_ls = np.array(src_flat_label_ls)
    #     src_col_repr = torch.zeros_like(self.model.src_agents_layer._agents).to(self.device)
    #     for i in range(len(self.src_trn_ds.id2label)):
    #         src_col_repr[i, :] = torch.tensor(np.mean(src_final_repr[src_flat_label_ls == i], axis=0))
    #     self.model.src_agents_layer._agents = torch.nn.Parameter(src_col_repr)
    #
    #     tgt_final_repr = np.array(tgt_final_repr)
    #     tgt_flat_label_ls = np.array(tgt_flat_label_ls)
    #     tgt_col_repr = torch.zeros_like(self.model.tgt_agents_layer._agents).to(self.device)
    #     for i in range(len(self.tgt_trn_ds.id2label)):
    #         tgt_col_repr[i:, ] = torch.tensor(np.mean(tgt_final_repr[tgt_flat_label_ls == i], axis=0))
    #     self.model.tgt_agents_layer._agents = torch.nn.Parameter(tgt_col_repr)
    #
    #     self.logger.info('initialization completed')

    def eval_dataset(self, dataloader, data_source, return_orig_logits=False):
        final_repr = []
        flat_label_ls = []
        with torch.no_grad():
            self.model.eval()
            for batch_ndx, batch_tup in enumerate(tqdm(dataloader)):
                inputs = batch_tup["data"]
                labels_ls = batch_tup["label"]

                input_ids = inputs["input_ids"].to(self.device, non_blocking=True)
                token_type_ids = inputs["token_type_ids"].to(
                    self.device, non_blocking=True
                )
                attention_mask = inputs["attention_mask"].to(
                    self.device, non_blocking=True
                )
                # model needs this id to filter out the output correspond to the [CLS] token
                cls_token_id = self.src_trn_ds.tokenizer.cls_token_id
                logits = self.model(
                    cls_token_id,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label_ls=labels_ls,
                    data_source=data_source,
                    return_logits=True,
                )
                final_repr.extend(logits.tolist())
                flat_label_ls.extend(
                    [
                        num
                        for sublist in labels_ls
                        for subsublist in sublist
                        for num in subsublist
                    ]
                )

        # put it back to train mode
        self.model.train()
        return final_repr, flat_label_ls

    def eval_to_match(self):
        self.logger.info(f"begin the eval-to-match process")

        src_dl = self.src_val_dl
        tgt_dl = self.tgt_val_dl

        src_final_repr, src_flat_label_ls = self.eval_dataset(src_dl, "src")
        tgt_final_repr, tgt_flat_label_ls = self.eval_dataset(tgt_dl, "tgt")

        src_final_repr = torch.tensor(src_final_repr)
        src_flat_label_ls = torch.tensor(src_flat_label_ls)
        tgt_final_repr = torch.tensor(tgt_final_repr)
        tgt_flat_label_ls = torch.tensor(tgt_flat_label_ls)

        src_counts = torch.bincount(src_flat_label_ls)
        tgt_counts = torch.bincount(tgt_flat_label_ls)
        assert (
            all(count == self.n_val_cols for count in src_counts)
            and len(src_counts) == self.src_df.shape[1]
        )
        assert (
            all(count == self.n_val_cols for count in tgt_counts)
            and len(src_counts) == self.src_df.shape[1]
        )

        #################################################################
        #################################################################
        # make sure the first row of it is corresponding to
        # the first column of the table, which means that the average embedding
        # of the embeddings with label 0 should be put into it.
        # Same for all others.
        src_centers = torch.empty(
            (self.src_df.shape[1], src_final_repr.shape[1]), dtype=src_final_repr.dtype
        )
        for label in range(self.src_df.shape[1]):
            label_mask = src_flat_label_ls == label
            label_embeddings = src_final_repr[label_mask]
            center_embedding = torch.mean(label_embeddings, dim=0)
            center_embedding = F.normalize(center_embedding, dim=0)
            src_centers[label] = center_embedding

        tgt_centers = torch.empty(
            (self.tgt_df.shape[1], tgt_final_repr.shape[1]), dtype=tgt_final_repr.dtype
        )
        for label in range(self.tgt_df.shape[1]):
            label_mask = tgt_flat_label_ls == label
            label_embeddings = tgt_final_repr[label_mask]
            center_embedding = torch.mean(label_embeddings, dim=0)
            center_embedding = F.normalize(center_embedding, dim=0)
            tgt_centers[label] = center_embedding

        # update the similarity matrix used to obtain the matching results
        self.curr_sim_matrix = torch.matmul(src_centers, tgt_centers.T)

        #################################################################
        #################################################################

        if not (
            len(src_flat_label_ls) == self.src_df.shape[1]
            or len(tgt_flat_label_ls) == self.tgt_df.shape[1]
        ):
            # not in one-shot prediction mode
            self.src_val_silhouette_score = silhouette_score(
                src_final_repr, src_flat_label_ls, metric="cosine"
            )
            self.tgt_val_silhouette_score = silhouette_score(
                tgt_final_repr, tgt_flat_label_ls, metric="cosine"
            )

            src_agents = self.model.src_agents_layer.agents.detach().cpu()
            tgt_agents = self.model.tgt_agents_layer.agents.detach().cpu()

            self.src_agent_sh_ls = silhouette_samples(
                torch.cat((src_final_repr, src_agents)),
                torch.cat(
                    (src_flat_label_ls, torch.tensor(list(range(src_agents.shape[0]))))
                ),
                metric="cosine",
            )[-src_agents.shape[0] :]
            self.src_center_sh_ls = silhouette_samples(
                torch.cat((src_final_repr, src_centers)),
                torch.cat(
                    (src_flat_label_ls, torch.tensor(list(range(src_agents.shape[0]))))
                ),
                metric="cosine",
            )[-src_agents.shape[0] :]
            self.tgt_agent_sh_ls = silhouette_samples(
                torch.cat((tgt_final_repr, tgt_agents)),
                torch.cat(
                    (tgt_flat_label_ls, torch.tensor(list(range(tgt_agents.shape[0]))))
                ),
                metric="cosine",
            )[-tgt_agents.shape[0] :]
            self.tgt_center_sh_ls = silhouette_samples(
                torch.cat((tgt_final_repr, tgt_centers)),
                torch.cat(
                    (tgt_flat_label_ls, torch.tensor(list(range(tgt_agents.shape[0]))))
                ),
                metric="cosine",
            )[-tgt_agents.shape[0] :]

    def get_matches(self, top_k=10):
        src_table_name = "source"
        tgt_table_name = "target"
        matches = dict()
        for i in range(self.curr_sim_matrix.shape[0]):
            for j in range(self.curr_sim_matrix.shape[1]):
                matches[
                    (
                        (src_table_name, self.src_trn_ds.id2label[i]),
                        (tgt_table_name, self.tgt_trn_ds.id2label[j]),
                    )
                ] = self.curr_sim_matrix[i][j].item()

        curr_sorted_matches = {
            k: v
            for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)
        }
        # (('source', 'Clin_Stage_Dist_Mets-cM'), ('target', 'revision')): -0.24113240838050842

        # ================== get top-k =====================
        top_k_matches = {}
        for k, v in curr_sorted_matches.items():
            source_col = k[0][1]
            target_col = k[1][1]

            if source_col not in top_k_matches:
                top_k_matches[source_col] = [(target_col, v)]
            else:
                if len(top_k_matches[source_col]) >= top_k:
                    continue
                top_k_matches[source_col].append((target_col, v))

        cleaned_sorted_matches = {}
        for source_col, target_cols in top_k_matches.items():
            for target_col, score in target_cols:
                cleaned_sorted_matches[
                    (("source", source_col), ("target", target_col))
                ] = score
        cleaned_sorted_matches = {
            k: v
            for k, v in sorted(
                cleaned_sorted_matches.items(), key=lambda item: item[1], reverse=True
            )
        }

        return cleaned_sorted_matches

    def update_curr_match_status(self):
        src_table_name = "source"
        tgt_table_name = "target"
        matches = dict()
        for i in range(self.curr_sim_matrix.shape[0]):
            for j in range(self.curr_sim_matrix.shape[1]):
                matches[
                    (
                        (src_table_name, self.src_trn_ds.id2label[i]),
                        (tgt_table_name, self.tgt_trn_ds.id2label[j]),
                    )
                ] = self.curr_sim_matrix[i][j].item()
        self.curr_sorted_matches = {
            k: v
            for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)
        }

        self.logger.critical(
            f"curr_sorted_matches----------->{self.curr_sorted_matches}"
        )
        # when in benchmark testing
        if self.golden_matches is not None:
            self.curr_metrics = self.get_metrics(self.curr_sorted_matches)

            # record the  numerical column performance
            top_ground_truth_matches = list(
                map(lambda m: frozenset(m), list(self.curr_sorted_matches.keys()))
            )
            top_ground_truth_matches = top_ground_truth_matches[
                : len(self.golden_matches.expected_matches)
            ]

            self.logger.critical(
                f"top_ground_truth_matches----------->{top_ground_truth_matches}"
            )
            for match in self.numerical_golden_matches_set:
                if match in top_ground_truth_matches:
                    self.curr_numerical_matches_set.add(match)

    def compute_batch(self, src_batch, tgt_batch, total_iter_len, curr_iter):
        src_inputs = src_batch["data"]
        src_labels_ls = src_batch["label"]

        tgt_inputs = tgt_batch["data"]
        tgt_labels_ls = tgt_batch["label"]

        src_input_ids = src_inputs["input_ids"].to(self.device, non_blocking=True)
        src_token_type_ids = src_inputs["token_type_ids"].to(
            self.device, non_blocking=True
        )
        src_attention_mask = src_inputs["attention_mask"].to(
            self.device, non_blocking=True
        )
        src_labels_ls = [
            [
                torch.tensor(labels, dtype=torch.long).to(
                    self.device, non_blocking=True
                )
                for labels in labels_ls
            ]
            for labels_ls in src_labels_ls
        ]

        tgt_input_ids = tgt_inputs["input_ids"].to(self.device, non_blocking=True)
        tgt_token_type_ids = tgt_inputs["token_type_ids"].to(
            self.device, non_blocking=True
        )
        tgt_attention_mask = tgt_inputs["attention_mask"].to(
            self.device, non_blocking=True
        )
        tgt_labels_ls = [
            [
                torch.tensor(labels, dtype=torch.long).to(
                    self.device, non_blocking=True
                )
                for labels in labels_ls
            ]
            for labels_ls in tgt_labels_ls
        ]

        # model needs this id to filter out the output correspond to the [CLS] token
        cls_token_id = self.src_trn_ds.tokenizer.cls_token_id

        src_match_loss, src_self_assign_loss = self.model(
            cls_token_id,
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            token_type_ids=src_token_type_ids,
            label_ls=src_labels_ls,
            data_source="src",
            return_logits=False,
        )

        tgt_match_loss, tgt_self_assign_loss = self.model(
            cls_token_id,
            input_ids=tgt_input_ids,
            attention_mask=tgt_attention_mask,
            token_type_ids=tgt_token_type_ids,
            label_ls=tgt_labels_ls,
            data_source="tgt",
            return_logits=False,
        )

        sim_matrix, opt_trans_sim_matrix = self.model.get_sim_matrix_with_recloss()

        orig_sim_rec_loss = torch.nn.functional.cross_entropy(
            sim_matrix.view(-1), opt_trans_sim_matrix.view(-1)
        )
        # could see it as a regularization on a part of the parameters of the model
        sim_rec_loss = orig_sim_rec_loss

        ##### 6 linear increase
        num_warmup_steps = 0.3 * total_iter_len
        if curr_iter <= num_warmup_steps:
            factor = curr_iter / num_warmup_steps
            sim_recloss_weight = factor * self.rec_loss_weight
        else:
            sim_recloss_weight = self.rec_loss_weight

        self.curr_sim_matrix = sim_matrix.detach().clone().cpu()

        # losses with different weight
        loss = (
            src_match_loss * self.meta_match_loss_weight
            + tgt_match_loss * self.meta_match_loss_weight
            + src_self_assign_loss * self.agent_delegate_loss_weight
            + tgt_self_assign_loss * self.agent_delegate_loss_weight
            + sim_rec_loss * sim_recloss_weight
        )

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

    def do_trn(self, process_mode=0):
        self.model.train()
        for batch_ndx, (src_batch, tgt_batch) in enumerate(
            tqdm(zip(self.src_trn_dl, self.tgt_trn_dl), total=len(self.src_trn_dl)),
            start=1,
        ):
            self.compute_batch(src_batch, tgt_batch, len(self.src_trn_dl), batch_ndx)

            trained_samples = batch_ndx * self.batch_size

            if trained_samples >= self.check_point_sample_ls[0]:
                self.check_point_cols_ls.pop(0)
                self.check_point_sample_ls.pop(0)

                if process_mode == 0:
                    # the normal train to match mode, the similarity matrix is updated in compute_batch function
                    self.update_curr_match_status()
                    self.record_middle_info()

                if len(self.check_point_cols_ls) == 0 and process_mode == 0:
                    # if it's the final point, save the records to a file
                    self.save_final_info()

    def init_metrics(self):
        metric_expected = [
            "recall_at_sizeof_ground_truth",
            # "precision",
            # "recall",
            # "f1_score",
            # "precision_at_n_percent",
        ]
        metric_fns = [getattr(module_metric, met) for met in metric_expected]
        return metric_fns

    def get_metrics(self, matches):
        final_metrics = dict()
        for metric in self.metric_fns:
            if metric.__name__ != "precision_at_n_percent":
                if metric.__name__ in ["precision", "recall", "f1_score"]:
                    # only filter with the 'one-to-one' when it's these three metric
                    final_metrics[metric.__name__] = metric(
                        matches, self.golden_matches, True
                    )
                else:
                    final_metrics[metric.__name__] = metric(
                        matches, self.golden_matches
                    )
            else:
                for n in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                    final_metrics[
                        metric.__name__.replace("_n_", "_" + str(n) + "_")
                    ] = metric(matches, self.golden_matches, n)
        return final_metrics

    def record_middle_info(self):
        # Run this function only if necessary to avoid impacting efficiency.
        self.res_dic["dataname"] = self.dataset_name

        # Not in benchmark mode, the following is not needed
        if self.golden_matches is None:
            return

        if "metrics" not in self.res_dic:
            self.res_dic["metrics"] = {k: [v] for k, v in self.curr_metrics.items()}
        else:
            for k, v in self.curr_metrics.items():
                self.res_dic["metrics"][k].append(v)

        if "n_matches" not in self.res_dic:
            self.res_dic["n_matches"] = {
                "n_generated_matches": [
                    round(
                        len(self.golden_matches.expected_matches)
                        * self.curr_metrics["recall_at_sizeof_ground_truth"]
                    )
                ],
                "n_expected_matches": [len(self.golden_matches.expected_matches)],
            }
        else:
            self.res_dic["n_matches"]["n_generated_matches"].append(
                round(
                    len(self.golden_matches.expected_matches)
                    * self.curr_metrics["recall_at_sizeof_ground_truth"]
                )
            )
            self.res_dic["n_matches"]["n_expected_matches"].append(
                len(self.golden_matches.expected_matches)
            )

        if "n_num_matches" not in self.res_dic:
            self.res_dic["n_num_matches"] = {
                "n_num_generated_matches": [len(self.curr_numerical_matches_set)],
                "n_num_expected_matches": [len(self.numerical_golden_matches_set)],
            }
        else:
            self.res_dic["n_num_matches"]["n_num_generated_matches"].append(
                len(self.curr_numerical_matches_set)
            )
            self.res_dic["n_num_matches"]["n_num_expected_matches"].append(
                len(self.numerical_golden_matches_set)
            )

        src_val_silhouette_score = (
            float(self.src_val_silhouette_score)
            if self.src_val_silhouette_score is not None
            else self.src_val_silhouette_score
        )
        tgt_val_silhouette_score = (
            float(self.tgt_val_silhouette_score)
            if self.tgt_val_silhouette_score is not None
            else self.tgt_val_silhouette_score
        )
        if "src_val_silhouette_score" not in self.res_dic:
            self.res_dic["src_val_silhouette_score"] = [src_val_silhouette_score]
            self.res_dic["tgt_val_silhouette_score"] = [tgt_val_silhouette_score]
        else:
            self.res_dic["src_val_silhouette_score"].append(src_val_silhouette_score)
            self.res_dic["tgt_val_silhouette_score"].append(tgt_val_silhouette_score)

        if "run_time" not in self.res_dic:
            self.res_dic["run_time"] = {
                "total_time": [time.time() - self.algo_begin_time],
                "algorithm_time": [time.time() - self.algo_comp_begin_time],
            }
        else:
            self.res_dic["run_time"]["total_time"].append(
                time.time() - self.algo_begin_time
            )
            self.res_dic["run_time"]["algorithm_time"].append(
                time.time() - self.algo_comp_begin_time
            )

        if self.device.type == "cuda":
            cuda_max_allocated = torch.cuda.max_memory_allocated(device=self.device)
            cuda_max_reserved = torch.cuda.max_memory_reserved(device=self.device)
        else:
            cuda_max_allocated = 0
            cuda_max_reserved = 0
        self.res_dic["cuda_max_allocated"] = cuda_max_allocated
        self.res_dic["cuda_max_reserved"] = cuda_max_reserved

        # self.res_dic['parameters'] = self.running_params
        # self.res_dic['parameters'] = loc_running_params
        # self.res_dic['parameters'] = [x for x in self.running_params if x.startswith('comment=')]
        self.res_dic["parameters"] = self.comment

        self.res_dic["src_center_sh_ls"] = (
            self.src_center_sh_ls.tolist()
            if self.src_center_sh_ls is not None
            else None
        )
        self.res_dic["src_agent_sh_ls"] = (
            self.src_agent_sh_ls.tolist() if self.src_agent_sh_ls is not None else None
        )
        self.res_dic["tgt_center_sh_ls"] = (
            self.tgt_center_sh_ls.tolist()
            if self.tgt_center_sh_ls is not None
            else None
        )
        self.res_dic["tgt_agent_sh_ls"] = (
            self.tgt_agent_sh_ls.tolist() if self.tgt_agent_sh_ls is not None else None
        )

    def save_final_info(self):
        if len(self.res_dic) == 0:
            # there are no intermediate records in eval-to-match mode, do it here
            self.record_middle_info()

        if self.store_matches:
            bm = {str(key): value for key, value in self.curr_sorted_matches.items()}
            self.res_dic["matches"] = bm

        output_dir = self.get_output_dir()
        run_id = "-".join([d for d in self.dataset_name.split("/")])
        res_file = os.path.join(output_dir, f"{run_id}.json")
        if os.path.dirname(res_file):
            os.makedirs(os.path.dirname(res_file), exist_ok=True)
        with open(res_file, "w") as f:
            json.dump(self.res_dic, f, indent=2)

    def main(self):
        if self.process_mode == 0:
            self.do_trn(self.process_mode)
        elif self.process_mode == 1:
            self.do_trn(self.process_mode)
            self.eval_to_match()
            self.update_curr_match_status()
            self.save_final_info()
        elif self.process_mode == 2:
            self.eval_to_match()
            self.update_curr_match_status()
            self.save_final_info()
        else:
            print("wrong --process-mode parameter")
            sys.exit()


if __name__ == "__main__":
    TrainApp().main()
