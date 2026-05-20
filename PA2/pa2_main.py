import argparse
import time
import traceback
import importlib
import os
import getpass
from utilities import spark_init
from utilities import PA2Test
from utilities import PA2Data
from utilities import TASK_NAMES
from utilities import data_cat
import databricks.koalas as ks

class PA2Executor(object):
    def __init__(
        self,
        args,
        task_imls=None,
        input_format='dataframe',
        synonmys=['piano', 'rice', 'laptop'],
        output_pid_folder=False
    ):

        self.spark = spark_init(args.pid)
        if input_format == 'koalas':
            ks.set_option('compute.default_index_type', 'distributed')
        path_dict = {
            'review': args.review_filename,
            'product': args.product_filename,
            'product_processed': args.product_processed_filename,
            'ml_features_train': args.ml_features_train_filename,
            'ml_features_test': args.ml_features_test_filename
            }

        self.task_imls = task_imls
        self.tests = PA2Test(self.spark, args.test_results_root)
        if output_pid_folder:
            output_root = os.path.join(args.output_root, args.pid)
        else:
            output_root = args.output_root
        self.data_io = PA2Data(self.spark, path_dict, output_root, 
                               deploy=True, input_format=input_format)

        self.data_dict, self.count_dict = self.data_io.load_all(
            input_format=input_format, no_cache=True)
        self.task_names = TASK_NAMES
        self.synonmys = synonmys
        


    def arguments(self):
        arguments = {
            "task_1": [self.data_io, self.data_dict['review'], self.data_dict['product']],
            "task_2": [self.data_io, self.data_dict['product']],
            "task_3": [self.data_io, self.data_dict['product']],
            "task_4": [self.data_io, self.data_dict['product']],
            "task_5": [self.data_io, self.data_dict['product_processed']] + self.synonmys,
            "task_6": [self.data_io, self.data_dict['product_processed']],
            "task_7": [self.data_io, self.data_dict['ml_features_train'], self.data_dict['ml_features_test']],
            "task_8": [self.data_io, self.data_dict['ml_features_train'], self.data_dict['ml_features_test']]
            
        }
        return arguments

    def tasks(self):
        tasks = {
            "task_1": self.task_imls.task_1,
            "task_2": self.task_imls.task_2,
            "task_3": self.task_imls.task_3,
            "task_4": self.task_imls.task_4,
            "task_5": self.task_imls.task_5,
            "task_6": self.task_imls.task_6,
            "task_7": self.task_imls.task_7,
            "task_8": self.task_imls.task_8
        }
        return tasks

    def eval(self):
        results = []
        timings = []
        begin = time.time()
        for part in ['part_1', 'part_2']:
            print ("Running {} ...".format(part))
            self.data_dict, self.count_dict = self.data_io.cache_switch(self.data_dict, part)
            results_part, timings_part = self.eval_by_part(part)
            results += results_part
            timings += timings_part            
        e2e_dur = time.time()-begin
        print ("End to end time (including data io): {} sec".format(e2e_dur))
        print ("End to end time (excluding data io): {} sec".format(sum(timings)))
        timings.append(e2e_dur)
        return results, timings

    def eval_one(self, task, fargs, task_name):
        result = False
        try:
            res = task(*fargs)
            result = self.tests.test(res, task_name)
        except Exception as e:
            print(
                "{} failed to execute, please inspect your code before submission. Exception: {}" \
                .format(task_name, e)
                )
            traceback.print_exc()
        return result

    def eval_by_part(self, part):
        results = []
        timings = []
        if part == 'part_1':
            task_names = self.task_names[:6]
        elif part == 'part_2':
            task_names = self.task_names[6:]
        for task_name in task_names:
            result, sub_task_dur = self.eval_by_name(task_name)
            results.append(result)
            timings.append(sub_task_dur)
            print ("{} time: {} sec".format(task_name, sub_task_dur))
        return results, timings
        
    def eval_by_name(self, task_name):
        arguments, tasks = self.arguments(), self.tasks()
        fargs = arguments[task_name]
        task = tasks[task_name]
        sub_task_begin = time.time()
        result = self.eval_one(task, fargs, task_name)
        sub_task_end = time.time()
        sub_task_dur = sub_task_end - sub_task_begin
        return result, sub_task_dur

def get_main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--review_filename', type=str, default=data_cat.review_filename
    )
    parser.add_argument(
        '--product_filename', type=str, default=data_cat.product_filename
    )
    parser.add_argument(
        '--product_processed_filename', type=str,
        default=data_cat.product_processed_filename
    )
    parser.add_argument(
        '--ml_features_train_filename', type=str,
        default=data_cat.ml_features_train_filename
    )
    parser.add_argument(
        '--ml_features_test_filename', type=str,
        default=data_cat.ml_features_test_filename
    )
    parser.add_argument(
        '--test_results_root', type=str,
        default=data_cat.test_results_root
    )
    parser.add_argument(
        '--pid', type=str
    )
    parser.add_argument(
        '--username', type=str
    )
    parser.add_argument(
        '--module_name', type=str,
        default='assignment2'
    )
    parser.add_argument(
        '--output_root', type=str,
        default=None
    )
    parser.add_argument('--synonmys', nargs='+', type=str, default=['piano', 'rice', 'laptop'])
    return parser
        
if __name__ == "__main__":

    parser = get_main_parser()
    args = parser.parse_args()
    #username = getpass.getuser()
    if not args.output_root:
        args.output_root = '/home/{}/{}-pa2/test_results'.format(
            args.username, args.pid)
    task_imls = importlib.import_module(args.module_name)
    pa2 = PA2Executor(args, task_imls, task_imls.INPUT_FORMAT, args.synonmys)
    results, timings = pa2.eval()
    res = []
    for task_name, result, timing in zip(TASK_NAMES, results, timings):
        res.append({'task_name': task_name,
                   'passed': result, 'time_sec': timing})
    pa2.data_io.save(res, 'summary')
