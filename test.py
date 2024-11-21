# from datetime import datetime
# # 获取当前日期时间
# now = datetime.now()
# # 获取当前星期几的数字 (0 表示周一，6 表示周日)
# day_of_week_number = now.weekday()
# # 将数字映射为中文的星期几
# weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
# # 输出中文的星期几
# print(weekdays[day_of_week_number])

from datetime import datetime
# 输入时间字符串
time_str = "2024-09-24 13:47:14"
# 解析时间字符串
dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
# 获取星期几的数字 (0 表示周一，6 表示周日)
day_of_week_number = dt.weekday()
# 将数字映射为中文的星期几
weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
# 输出中文的星期几
print(weekdays[day_of_week_number])

