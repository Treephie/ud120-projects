#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    # print ages
    # print type(ages)
    
    # cleaned_data = []
    #
    # data_list = []
    # ### your code goes here
    # n = len(ages)
    # for i in range(n):
    #     age = ages[i][0]
    #     net_worth = net_worths[i][0]
    #     pred_net_worth = predictions[i][0]
    #     error = pred_net_worth - net_worth
    #
    #     data = (age, net_worth, error)
    #     data_list.append(data)
    #
    # # sort
    # sorted_data_list = sorted(data_list, key=lambda x: (x[0], x[1], -x[2]))
    # print sorted_data_list[1]
    #
    # # remove 10%
    # remain_num = int(n * 0.9)
    # for x in range(remain_num):
    #     cleaned_data.append(sorted_data_list[x])
    #
    # print cleaned_data[1]



    cleaned_data = []

    ### your code goes here
    errors = []
    for index in range(0, len(predictions)):
        errors.append(predictions[index] - net_worths[index])
    largest_ten_percent_indices = sorted(range(len(errors)), key=lambda i: errors[i])[-len(predictions)/10:]

    for index in range(0, len(predictions)):
        if index not in largest_ten_percent_indices:
            cleaned_data.append((ages[index], net_worths[index], errors[index]))


    return cleaned_data
