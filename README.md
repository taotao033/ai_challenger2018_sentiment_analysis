Introductions:
dataset dowload link:https://challenger.ai

data_preprocess.py : There are many methods to process data.

classifier_1_svm.py : binary classification {0(-2), 1(-1,0,1)}.

classifier_2_svm.py : binary classification {0(-1,0), 1(1)}.

classifier_3_svm.py : binary classification {-1,0}.

evaluate.py : use validationset to evaluate these models,generate
validation files accordding to samples format.

f1_score.py : calculate F1_measure.

predict_20label.py : predict testaset and submit.

																	2018 AIChallenger 全球AI挑战赛

																	## 数据标签分布统计

																	|class|-2|-1|0|1|
																	|---|---|---|---|---|
																	|location_traffic_convenience|81382|1318|1046|21254|
																	|location_distance_from_business_district|83680|586|533|20201|
																	|location_easy_to_find|80605|3976|2472|17947|
																	|service_wait_time|92763|3034|4382|4821|
																	|service_waiters_attitude|42410|8684|12534|41372|
																	|service_parking_convenience|98276|1323|1456|3945|
																	|service_serving_speed|88700|5487|2379|8434|
																	|price_level|52820|12375|24249|15556|
																	|price_cost_effective|80242|3011|3072|18675|
																	|price_discount|64243|1716|18255|20786|
																	|environment_decoration|53916|2139|9492|39453|
																	|environment_noise|73445|3077|4843|23635|
																	|environment_space|65398|5706|9262|24634|
																	|environment_cleaness|66598|4513|4703|29186|
																	|dish_portion|56917|10018|9506|28559|
																	|dish_taste|5070|4363|40200|55367|
																	|dish_look|75975|3178|4675|21172|
																	|dish_recommendation|84767|2275|1988|15970|
																	|others_overall_experience|2110|9384|23436|70070|
																	|others_willing_to_consume_again|65600|4159|2913|32328|

<div align="center">
<img src="https://github.com/taotao033/ai_challenger2018_sentiment_analysis/blob/master/images/Figure_1_number_of_comments_per_category.png" width="800" height="532" align=center/>
<img src="https://github.com/taotao033/ai_challenger2018_sentiment_analysis/blob/master/images/Figure_2_Multiple_categories_per_comment.png" width="800" height="500" align=center/>
</div>
