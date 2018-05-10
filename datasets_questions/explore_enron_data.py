#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "dataset size:", len(enron_data)
# print "no. of feature:", len(enron_data['METTS MARK'])
# print "features:", enron_data["SKILLING JEFFREY K"]
#
enron_data_poi = [x for x in enron_data if enron_data[x]["poi"] == 1]
print "nu. of poi:", len(enron_data_poi)
#
# print "total stock value of James Prentice:", enron_data["PRENTICE JAMES"]["total_stock_value"]
# print "emails from Wesley Colwell to poi:", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
# print "exercised_stock_options of Jefferey Skilling:", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
#
# print enron_data["SKILLING JEFFREY K"]["total_payments"], enron_data["LAY KENNETH L"]["total_payments"], enron_data["FASTOW ANDREW S"]["total_payments"]

enron_data_sal = [x for x in enron_data if enron_data[x]["total_payments"] == "NaN"]
print len(enron_data_sal)

# enron_data_email = [x for x in enron_data if enron_data[x]["email_address"] != "NaN"]
# print len(enron_data_email)

# what percentage of total_payments have "NaN"
enron_data_nan = [x for x in enron_data if enron_data[x]["total_payments"] == "NaN"]
print len(enron_data_nan)/len(enron_data)

# what percentage of POIs total_payments have "NaN"
enron_data_nan_poi = [x for x in enron_data if enron_data[x]["poi"] == 1 and enron_data[x]["total_payments"] == "NaN"]
print len(enron_data_nan_poi)/len(enron_data_poi)
