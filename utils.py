from pyspark.sql import SparkSession
import json


def get_spark_session(appName):
    spk = SparkSession. \
        builder. \
        appName('test'). \
        getOrCreate()
    return spk


def get_config_data():
    with open("config.json", 'r') as config_file:
        config = json.load(config_file)
    return config


def load_data_from_csv(spark, file_path):
    """
    Loads data csv file's path provided
    :param file_path: csv file path
    :return: Returns the only distinct data removing duplicates
    """
    data = spark.read.format('csv').options(Header=True).load(file_path)
    return data.distinct()


def store_output_to_csv(output, file_path):
    output.write.format('csv').mode('overwrite').save(file_path, header=True)
