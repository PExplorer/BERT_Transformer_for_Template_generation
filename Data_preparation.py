import pandas as pd
import re 

src_file_list = ['src_train.txt','src_valid.txt','src_test.txt']
target_file_list = ['train.txt','valid.txt','test.txt']
file_name_list = ['train_data','validation_data','test_data']

for j in range(3):
	src_file = src_file_list[i]
	target_file = target_file_list[i]
	file_name = file_name_list[i]
	
	with open(target_file,"r") as ins:
	    target_array=[]
		for line in ins:
			target_array.append(line)
			
    with open(src_file,"r"):
	    src_array = []
		for line in ins:
		    src_array.append(line)
			
	data = []
	
	for i in range(len(src_array)):
		name = re.search('__start_name__(.*)__end_name__',src_array[i])
		eat_type = re.search('__start_eatType__(.*)__end_eatType__',src_array[i])
		food = re.search('__start_food__(.*)__end_food__',src_array[i])
		price_range= re.search('__start_priceRange__(.*)__end_priceRange__',src_array[i])
		customer_rating = re.search('__start_customerRating__(.*)__end_customerRating__',src_array[i])
		area = re.search('__start_area__(.*)__end_area__',src_array[i])
		family_freindly = re.search('__start_familyFriendly__(.*)__end_familyFriendly__',src_array[i])
		near_location = re.search('__start_near__(.*)__end_near__',src_array[i])
		
		name_value = "unknown"
		eat_type_value = "unknown"
		food_value = "unknown"
		price_range_value = "unknown"
		customer_rating_value = "unknown"
		area_value = "unknown"
		family_freindly_value = "unknown"
		near_location_value ="unknown"
		
		if str(type(name))!="<class 'NoneType'>":
		    name_value = name.group(1)
		if str(type(eat_type))!="<class 'NoneType'>":
		    eat_type_value = eat_type.group(1)
        if str(type(food))!="<class 'NoneType'>":
		    food_value = food_type.group(1)
        if str(type(price_range_value))!="<class 'NoneType'>":
		    price_range_value = price_range.group(1)
        if str(type(customer_rating))!="<class 'NoneType'>":
		    customer_rating_value = customer_rating.group(1)
        if str(type(area))!="<class 'NoneType'>":
		    area_value = area.group(1)
        if str(type(family_freindly))!="<class 'NoneType'>":
		    family_freindly_value = family_freindly.group(1)
        if str(type(near_location))!="<class 'NoneType'>":
		    near_location_value = name_value.group(1)
        
		if j==2:
		   target = "NA"
		else:
		   target = re.match("(.*?)<eos>", target_array[i]).group()
		   
		data.append({"1 name":"Restaurant name" + " " + name_value,
		             "2 eat_type":"eat type" + " " + eat_type_value,
					 "3 food":"food type" + " " + food_value,
					 "4 Price range":"price range" + " " + price_range_value,
					 "5 Customer rating":"Customer rating" + " " + customer_rating_value,
					 "6 Area":"Area" + " " + area_value,
					 "7 Family friendly":"Family friendly " + " " + family_freindly_value,
					 "8 Near location":"Near location" + " " + near_location_value,
					 "91 target":"Restaurant name" + " " + name_value
					 })
					 
		data_df = pd.DataFrame(data)
		
		def concatenate_text(row):
		    row_combined = " "
			for value in row[:8]:
                if "Unknown" not in value:
                    row_combined = row_cmobined + " " + value 
            return row_combined 

        def remove_words(row):
            if row["91 target"]!="NA":
                target_sent = row["91 target"]
                target_sent = target_sent.replace('. <eos>','.')
                target_sent = target_sent.replace('<eos>','.')
			else:
			    target_sent = "NA"
			return target_sent
			
		data_df["input_text"] = data_df.apply(lambda row:concatenate_text(row),axis=1)
		data_df["target_text"] = data_df.apply(lambda row:remove_words(row),axis=1)
		
		data_df1 = data_df[["target_text","input_text"]]
		filename1 = file_name + "_full" + ".csv"
		filename2 = file_name+ ".csv"
		
		data_df.to_csv(filename1)
		data_df1.to_csv(filename2)
		
		
