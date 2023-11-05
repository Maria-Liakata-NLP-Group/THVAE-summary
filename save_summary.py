import os
import re
import pandas as pd
import errno


def save_summary():
    dir = "Timeline_dataset/timeline_summary"
    files = os.listdir(dir)
    summary_total = list()
    opinion_stroy = ''

    for file in files:
        if file == '.DS_Store':
            continue
        count = 0
        from_path = os.path.join(dir, file)
        id = file.split('.')[0]
        with open(from_path, 'r') as f:
            data = pd.read_csv(f, sep='\t')
            # if file == '58826_170.tsv':
            col = data['prompt']
            revs = data['review_text']
            summary = {}
            summary["prod_id"] = id
            summary["cat"] = 'reddit'
            for pro, rev in zip(col, revs):
                print(pro)
                if not isinstance(pro, str):
                    continue
                rev_len = len(pro.split())
                if rev_len < 2:
                    continue
                elif count < 8:
                    count += 1
                    summary['rev' + str(count)] = rev
                    summary['pro' + str(count)] = pro
                    opinion_stroy = pro
            if count >= 8:
                summary["summ1"] = opinion_stroy
                # print(opinion_stroy)
                summary_total.append(summary)

    summary_path = os.path.join('timeline_summary/', 'test.csv')
    write_group_to_csv(summary_path, summary_total, sep='\t')





def saf_mkdir(file_path):
    "create folders associated with host of the file"
    if os.path.dirname(file_path) and not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def write_group_to_csv(out_file_path, units, sep='\t'):
    """Writes data units into a CSV file.

        Args:
            out_file_path (str): self-explanatory.
            units (list): list with dicts (review texts and other attributes).
            sep (str): separation in the output csv files.

        Returns: None.

        """
    saf_mkdir(out_file_path)
    with open(out_file_path, 'w',encoding='utf-8') as f:
        header = None
        for du in units:
            if header is None:
                header = du.keys()
                f.write(sep.join(header) + "\n")
            str_to_write = sep.join([str(du[attr]) for attr in header])
            f.write(str_to_write + "\n")

def del_row():
    dir = "Timeline_dataset/timeline_test"
    files = os.listdir(dir)
    for file in files:
        if file == '.DS_Store':
            continue
        count = 0
        from_path = os.path.join(dir, file)
        with open(from_path, 'r') as f:
            print(file)
            data = pd.read_csv(f, sep='\t')
            col = data['prompt']
            for pro in col:
                print(pro)
                if not isinstance(pro, str):
                    print(pro, '+++++++++++++++')
                    print(col[count], '===============')
                    data = data.drop(data.index[count])
                    continue
                rev_len = len(pro.split())
                if rev_len < 2:
                    data = data.drop(data.index[count])
                    continue
                print(count)
                count += 1
        data.to_csv(from_path, index=False, sep='\t')





if __name__ == '__main__':

    # save_summary()
    del_row()