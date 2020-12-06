import os
import numpy as np
import csv


class GetTrainingTimeTraffic(object):
    def __init__(self, date_to_retrieve='YY_mm_dd_HH_MM_SS', target_output="algorithm.csv"):
        if date_to_retrieve != 'YY_mm_dd_HH_MM_SS':
            self.date_to_retrieve = date_to_retrieve  # Date must be in format %Y_%m_%d_%H_%M_%S
        else:
            print("Invalid date")
            exit(1)
        self.target_output = target_output

    def run(self):

        directory = '../output/log/'
        log_file = directory + 'log_' + self.date_to_retrieve + '.log'

        print(log_file)

        # Each non empty line is a sent command
        # Command of power is substituted by episode finishing line
        # Minus last line that is the total time

        counter_line = -1
        with open(log_file) as f:
            for line in f:
                if len(line.strip()) != 0:  # Not empty lines
                    counter_line += 1
            last_line = line

        secs = float(last_line.split()[3])
        np.set_printoptions(formatter={'float': lambda output: "{0:0.4f}".format(output)})

        print("Total lines", counter_line)
        print("Last line", last_line)
        print("Seconds", secs)

        if not os.path.isfile(self.target_output):  # If file does not exist
            # Write header
            with open(self.target_output, mode='w') as output_file:
                output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                output_writer.writerow(['Date', 'Training_time', 'Sent_commands'])

        with open(self.target_output, mode="a") as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow([self.date_to_retrieve, secs, counter_line])


def get_data_algos_path(sarsa, sarsa_lambda, qlearning, path=None):

    for dat in sarsa:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output='path' + str(path) + '_sarsa.csv').run()

    for dat in sarsa_lambda:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output='path' + str(path) + '_sarsa_lambda.csv').run()

    for dat in qlearning:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output='path' + str(path) + '_qlearning.csv').run()


if __name__ == '__main__':
    # get_data_before_tuning_unique_path()

    target_path = 0
    sarsa_dates = [
        '2020_12_06_16_51_59',
        '2020_12_06_16_52_06',
        '2020_12_06_16_52_14',
        '2020_12_06_16_52_22',
        '2020_12_06_16_52_31',
        '2020_12_06_16_52_39',
        '2020_12_06_16_52_47',
        '2020_12_06_16_52_55',
        '2020_12_06_16_53_02',
        '2020_12_06_16_53_10',
    ]

    sarsa_lambda_dates = [
        '2020_12_06_16_53_18',
        '2020_12_06_16_53_26',
        '2020_12_06_16_53_35',
        '2020_12_06_16_53_43',
        '2020_12_06_16_53_55',
        '2020_12_06_16_54_09',
        '2020_12_06_16_54_19',
        '2020_12_06_16_54_28',
        '2020_12_06_16_54_40',
        '2020_12_06_16_54_54',
    ]

    qlearning_dates = [
        '2020_12_06_16_50_38',
        '2020_12_06_16_50_46',
        '2020_12_06_16_50_55',
        '2020_12_06_16_51_03',
        '2020_12_06_16_51_11',
        '2020_12_06_16_51_19',
        '2020_12_06_16_51_27',
        '2020_12_06_16_51_35',
        '2020_12_06_16_51_43',
        '2020_12_06_16_51_51',
    ]

    get_data_algos_path(sarsa_dates, sarsa_lambda_dates, qlearning_dates, path=target_path)
