from utils import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = get_spark_session("Vehicle Crash data analysis")


class VehicleCrashAnalytics:
    """
    The Class VehicleCashAnalytics loads the data to be analyzed
    Individual Methods are defined to obtain and store the required Analytics result.
    """

    def __init__(self, tables_path):
        self.charges = load_data_from_csv(spark, tables_path["Charges"])
        self.endorsements = load_data_from_csv(spark, tables_path["Endorsements"])
        self.restrict = load_data_from_csv(spark, tables_path["Restrict"])
        self.damages = load_data_from_csv(spark, tables_path["Damages"])
        self.primary_person = load_data_from_csv(spark, tables_path["Primary Person"])
        self.unit = load_data_from_csv(spark, tables_path["Unit"])

    def analytics_1(self, output_path):
        """
        Ques : Find the number of crashes in which number of persons killed are male ?

        Approach : Filter the Primary_person table to find the data with MALE as the Person's
        Gender and person's injury severity is KILLED output file contains the distinct
        Crash_ID values in csv format.

        Stores the all Crash_ID's from filtered data to output path (includes duplicates).
        Displays the count of distinct number of crashes (accidents) in which number of
        persons killed are male.
        :param output_path- path to store the output
        :return None
        """
        filtered_data = self.primary_person.filter(
            (self.primary_person.PRSN_INJRY_SEV_ID.isin('KILLED'))
            & (self.primary_person.PRSN_GNDR_ID.isin('MALE')))
        output = filtered_data.select('CRASH_ID')
        store_output_to_csv(output, output_path)
        output.agg(F.count_distinct('CRASH_ID').alias('Analytics_1 Result')).show()

    def analytics_2(self, output_path):
        """
        Ques : How many two wheelers are booked for crashes?

        Approach : Since the data of Unit table contains duplicates remove duplicates rows.
        Considering a 2 Wheeler is any motorcycle or bike. Filter Unit data based on Vehicle
        body style description column VEH_BODY_STYL_ID.

        Stores all the filtered data to output path .
        Displays the count of 2 wheelers booked for crashes.
        :param output_path- path to store the output
        :return None
        """
        output = self.unit.filter(self.unit.VEH_BODY_STYL_ID.ilike('%MOTORCYCLE%'))
        store_output_to_csv(output, output_path)
        print("Analytics_2 Result :", output.count())

    def analytics_3(self, output_path):
        """
        Ques : Which state has highest number of crashes in which females are involved?

        Approach : Filter data with person's gender as FEMALE and Group by the State that
        issued Vehicle's driver license.

        Stores the data ranked based on highest Crash_count to output path.
        Displays the State with highest Crash_count value.
        :param output_path- path to store the output
        :return None
        """
        crash_count_data = self.primary_person.filter(self.primary_person.PRSN_GNDR_ID.isin('FEMALE')).groupBy(
            'DRVR_LIC_STATE_ID'). \
            agg(F.count_distinct('CRASH_ID').alias('Crash_count'))
        window = Window.orderBy(F.col("Crash_count").desc())
        ranked_data = crash_count_data.withColumn('Crash_count_rank', F.dense_rank().over(window))
        store_output_to_csv(ranked_data, output_path)
        ranked_data.filter(F.col('Crash_count_rank') == 1).select(
            F.col('DRVR_LIC_STATE_ID').alias('Analytics_3 Result')).show()

    def analytics_4(self, output_path):
        """
        Ques : Which are the Top 5th to 15th VEH_MAKE_IDs that contribute to a largest
        number of injuries including death

        Approach : Group the data based on Vehicle make ID and calculate the total count of
        injuries and death count. Apply rank to over the total count in descending order and
        filter it based on the rank.

        Stores the data ranked based on highest total injuries + death count to output path.
        Displays the VEH_MAKE_ID values for ranks from 5 to 15 from output.
        :param output_path- path to store the output
        :return None
        """
        ranked_data = self.unit.groupBy('VEH_MAKE_ID'). \
            agg(F.sum(self.unit.TOT_INJRY_CNT + self.unit.DEATH_CNT).alias('Total injuries+death'))
        window = Window.orderBy(F.col('Total injuries+death').desc())
        ranked_data = ranked_data.withColumn('Rank', F.dense_rank().over(window))
        output = ranked_data.filter((ranked_data.Rank >= 5) & (ranked_data.Rank <= 15))
        store_output_to_csv(output, output_path)
        output.select(F.col('VEH_MAKE_ID').alias('Analytics_4 Result')).show()

    def analytics_5(self, output_path):
        """
        Ques : For all the body styles involved in crashes, mention the top ethnic user
        group of each unique body style.


        Approach:
        Join the UNIT table data with PRIMARY PERSON on CRASH_ID and UNIT_NBR to and group
        it based on Vehicle body style& Person's Ethnic group to get Ethnic_count for each
        body style. Using Rank to get the top Ethnic group within each Vehicle body style.

        Stores and displays the data ranked based on highest Ethnic_count for each unique
        Vehicle body style.
        :param output_path- path to store the output
        :return None
        """
        join_condition = [self.primary_person.CRASH_ID == self.unit.CRASH_ID,
                          self.primary_person.UNIT_NBR == self.unit.UNIT_NBR]
        ethnic_count_data = self.primary_person.join(self.unit, join_condition, 'inner'). \
            groupBy('VEH_BODY_STYL_ID', 'PRSN_ETHNICITY_ID'). \
            agg(F.count('PRSN_ETHNICITY_ID').alias('Ethnic_count'))
        window = Window.partitionBy(F.col('VEH_BODY_STYL_ID')).orderBy(F.col('Ethnic_count').desc())
        output = ethnic_count_data.withColumn('Rank', F.dense_rank().over(window)).filter(F.col('Rank') == 1)
        store_output_to_csv(output, output_path)
        print("Analytics_5 Result")
        output.select(['VEH_BODY_STYL_ID', 'PRSN_ETHNICITY_ID']).show(truncate=False)

    def analytics_6(self, output_path):
        """
        Ques : Among the crashed cars, what are the Top 5 Zip Codes with highest number
        crashes with alcohols as the contributing factor to a crash (Use Driver Zip Code)

        Approach:
        Filter the Primary person data to get Driver data with a non null Zip code and
        filter Unit table data to get Car crash information with Alcohol as one of the
        contributing factors. Joining the filtered Car unit data and Driver data.
        Group the joined data based on Zip Code of driver to get crash_count for each
        Zip code.

        Stores and displays first 5 rows data ordered based on highest crash_count for each
        Zip Code.

        :param output_path- path to store the output
        :return None
        """
        driver_data = self.primary_person.filter(self.primary_person.PRSN_TYPE_ID.isin('DRIVER') &
                                                 self.primary_person.DRVR_ZIP.isNotNull())
        cars_data = self.unit.filter((self.unit.VEH_BODY_STYL_ID.ilike('%CAR%')) &
                                     (self.unit.CONTRIB_FACTR_1_ID.ilike('%ALCOHOL%')
                                      | self.unit.CONTRIB_FACTR_2_ID.ilike('%ALCOHOL%')
                                      | self.unit.CONTRIB_FACTR_P1_ID.ilike('%ALCOHOL%')))
        output = cars_data.join(driver_data, ['CRASH_ID', 'UNIT_NBR'], 'inner').groupby("DRVR_ZIP"). \
            count().orderBy(F.col('count').desc()).limit(5)
        store_output_to_csv(output, output_path)
        print("Analytics_6 Result")
        output.show()

    def analytics_7(self, output_path):
        """
        Ques : Count of Distinct Crash IDs where No Damaged Property was observed and
        Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance

        Approach:
        Filter the Unit table to get Cars with Insurance availed.Filter Damages table
        to check is Damaged property is NONE and with the insured cars data.Filter
        joined data where VEH_DMAG_SCL is above 4 using regular expression pattern.


        Stores and displays filtered result data.

        :param output_path- path to store the output
        :return None
        """
        insured_cars = self.unit.filter((self.unit.FIN_RESP_TYPE_ID.ilike('%INSURANCE%')) &
                                        self.unit.VEH_BODY_STYL_ID.ilike('%CAR%'))
        no_damage = self.damages.filter(self.damages.DAMAGED_PROPERTY.isin('NONE'))
        output = insured_cars.alias('ic').join(no_damage.alias('nd'),
                                               insured_cars.CRASH_ID == no_damage.CRASH_ID, 'inner'). \
            filter(insured_cars.VEH_DMAG_SCL_1_ID.rlike('DAMAGED [5-7].') |
                   insured_cars.VEH_DMAG_SCL_2_ID.rlike('DAMAGED [5-7].'))
        store_output_to_csv(output.select('ic.CRASH_ID'), output_path)
        output.agg(F.count_distinct('ic.CRASH_ID').alias("Analytics_7 Result")).show()

    def analytics_8(self, output_path):
        """
        Ques : : Determine the Top 5 Vehicle Makes where drivers are charged with speeding
         related offences, has licensed Drivers, used top 10 used vehicle colours and has
         car licensed with the Top 25 states with highest number of offences.

        Approach:
        Get top 10 Colors of Vehicles from Unit table. Get top 25 States with highest
        number of offences by joining charges with primary person table. Create a
        temporary table with filtered Primary person table to get crashes with
        Licensed drivers and also charged with speed offense. Join top 10 colors
        data with temporary table and followed by join with top 25 states.


        Stores and displays first 5 rows data ordered based on top Vehicle Makes ID's.

        :param output_path- path to store the output
        :return None
        """
        top10_colors = self.unit.groupBy(self.unit.VEH_COLOR_ID).count().orderBy('count', ascending=False).limit(10)
        join_condition = [self.primary_person.CRASH_ID == self.charges.CRASH_ID,
                          self.primary_person.UNIT_NBR == self.charges.UNIT_NBR]
        top25_state = self.charges.alias('od').join(self.primary_person, join_condition, 'inner').groupBy(
            'DRVR_LIC_STATE_ID'). \
            agg(F.count('od.CRASH_ID').alias('Offense_count')).orderBy('Offense_count', ascending=False).limit(25)
        license_driver = self.primary_person.filter(self.primary_person.DRVR_LIC_TYPE_ID.isin('DRIVER LICENSE'))
        speed_offense = license_driver.alias('ld').join(self.charges, 'CRASH_ID', 'inner').filter(
            self.charges.CHARGE.ilike('%SPEED%'))
        temp_data = self.unit.join(top10_colors, top10_colors.VEH_COLOR_ID == self.unit.VEH_COLOR_ID, 'inner')
        temp_data = temp_data.alias('td').join(speed_offense, [F.col('td.CRASH_ID') == F.col('ld.CRASH_ID'),
                                                               F.col('td.UNIT_NBR') == F.col('ld.UNIT_NBR')], 'inner')
        output = temp_data.join(top25_state, temp_data.DRVR_LIC_STATE_ID == top25_state.DRVR_LIC_STATE_ID, 'inner')
        output = output.groupBy('VEH_MAKE_ID').agg(F.count('VEH_MAKE_ID').alias('count')). \
            orderBy('count', ascending=False)
        store_output_to_csv(output, output_path)
        print("Analytics_8 Result")
        output.limit(5).show()


if __name__ == '__main__':
    config = get_config_data()
    vca = VehicleCrashAnalytics(config.get('InputTables'))
    output_paths = config.get('output_files')
    vca.analytics_1(output_paths["analytics_1"])
    vca.analytics_2(output_paths["analytics_2"])
    vca.analytics_3(output_paths["analytics_3"])
    vca.analytics_4(output_paths["analytics_4"])
    vca.analytics_5(output_paths["analytics_5"])
    vca.analytics_6(output_paths["analytics_6"])
    vca.analytics_7(output_paths["analytics_7"])
    vca.analytics_8(output_paths["analytics_8"])
