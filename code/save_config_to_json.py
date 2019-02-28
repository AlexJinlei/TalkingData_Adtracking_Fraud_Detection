import pandas as pd
import numpy as np
import json
import os

output_dir = '../data/input/'
config = 'feature_list.json'

#raw = ['raw_channel', 'raw_attributed_time', 'raw_ip', 'raw_app', 'raw_click_time', 'raw_click_time_int_seconds', 'raw_device', 'raw_is_attributed', 'raw_os', 'raw_click_id']

raw = ['raw_channel', 'raw_ip', 'raw_app', 'raw_device', 'raw_os']

#date_time = ['year', 'month', 'weekday', 'day', 'hour', 'minute', 'second', 'hour_of_day', 'hour_of_day_sin', 'hour_of_day_cos']

date_time = ['weekday', 'hour', 'minute', 'hour_of_day', 'hour_of_day_sin', 'hour_of_day_cos']

time_delta = ['ip_dt_forward', 'ip_dt_backward', 'app_dt_forward', 'app_dt_backward', 'device_dt_forward', 'device_dt_backward', 'os_dt_forward', 'os_dt_backward', 'channel_dt_forward', 'channel_dt_backward', 'ip_app_dt_forward', 'ip_app_dt_backward', 'ip_device_dt_forward', 'ip_device_dt_backward', 'ip_os_dt_forward', 'ip_os_dt_backward', 'ip_channel_dt_forward', 'ip_channel_dt_backward', 'app_device_dt_forward', 'app_device_dt_backward', 'app_os_dt_forward', 'app_os_dt_backward', 'app_channel_dt_forward', 'app_channel_dt_backward', 'device_os_dt_forward', 'device_os_dt_backward', 'device_channel_dt_forward', 'device_channel_dt_backward', 'os_channel_dt_forward', 'os_channel_dt_backward', 'ip_app_device_dt_forward', 'ip_app_device_dt_backward', 'ip_app_os_dt_forward', 'ip_app_os_dt_backward', 'ip_app_channel_dt_forward', 'ip_app_channel_dt_backward', 'ip_device_os_dt_forward', 'ip_device_os_dt_backward', 'ip_device_channel_dt_forward', 'ip_device_channel_dt_backward', 'ip_os_channel_dt_forward', 'ip_os_channel_dt_backward', 'app_device_os_dt_forward', 'app_device_os_dt_backward', 'app_device_channel_dt_forward', 'app_device_channel_dt_backward', 'app_os_channel_dt_forward', 'app_os_channel_dt_backward', 'device_os_channel_dt_forward', 'device_os_channel_dt_backward', 'ip_app_device_os_dt_forward', 'ip_app_device_os_dt_backward', 'ip_app_device_channel_dt_forward', 'ip_app_device_channel_dt_backward', 'ip_app_os_channel_dt_forward', 'ip_app_os_channel_dt_backward', 'ip_device_os_channel_dt_forward', 'ip_device_os_channel_dt_backward', 'app_device_os_channel_dt_forward', 'app_device_os_channel_dt_backward', 'ip_app_device_os_channel_dt_forward', 'ip_app_device_os_channel_dt_backward']

unique_count = ['unique_count_app_groupby_ip', 'unique_count_channel_groupby_ip', 'unique_count_device_groupby_ip', 'unique_count_os_groupby_ip', 'unique_count_channel_groupby_app', 'unique_count_device_groupby_app', 'unique_count_ip_groupby_app', 'unique_count_os_groupby_app', 'unique_count_app_groupby_device', 'unique_count_channel_groupby_device', 'unique_count_ip_groupby_device', 'unique_count_os_groupby_device', 'unique_count_app_groupby_os', 'unique_count_channel_groupby_os', 'unique_count_device_groupby_os', 'unique_count_ip_groupby_os', 'unique_count_app_groupby_channel', 'unique_count_device_groupby_channel', 'unique_count_ip_groupby_channel', 'unique_count_os_groupby_channel', 'unique_count_channel_groupby_ip_app', 'unique_count_device_groupby_ip_app', 'unique_count_os_groupby_ip_app', 'unique_count_app_groupby_ip_device', 'unique_count_channel_groupby_ip_device', 'unique_count_os_groupby_ip_device', 'unique_count_app_groupby_ip_os', 'unique_count_channel_groupby_ip_os', 'unique_count_device_groupby_ip_os', 'unique_count_app_groupby_ip_channel', 'unique_count_device_groupby_ip_channel', 'unique_count_os_groupby_ip_channel', 'unique_count_channel_groupby_app_device', 'unique_count_ip_groupby_app_device', 'unique_count_os_groupby_app_device', 'unique_count_channel_groupby_app_os', 'unique_count_device_groupby_app_os', 'unique_count_ip_groupby_app_os', 'unique_count_device_groupby_app_channel', 'unique_count_ip_groupby_app_channel', 'unique_count_os_groupby_app_channel', 'unique_count_app_groupby_device_os', 'unique_count_channel_groupby_device_os', 'unique_count_ip_groupby_device_os', 'unique_count_app_groupby_device_channel', 'unique_count_ip_groupby_device_channel', 'unique_count_os_groupby_device_channel', 'unique_count_app_groupby_os_channel', 'unique_count_device_groupby_os_channel', 'unique_count_ip_groupby_os_channel', 'unique_count_channel_groupby_ip_app_device', 'unique_count_os_groupby_ip_app_device', 'unique_count_channel_groupby_ip_app_os', 'unique_count_device_groupby_ip_app_os', 'unique_count_device_groupby_ip_app_channel', 'unique_count_os_groupby_ip_app_channel', 'unique_count_app_groupby_ip_device_os', 'unique_count_channel_groupby_ip_device_os', 'unique_count_app_groupby_ip_device_channel', 'unique_count_os_groupby_ip_device_channel', 'unique_count_app_groupby_ip_os_channel', 'unique_count_device_groupby_ip_os_channel', 'unique_count_channel_groupby_app_device_os', 'unique_count_ip_groupby_app_device_os', 'unique_count_ip_groupby_app_device_channel', 'unique_count_os_groupby_app_device_channel', 'unique_count_device_groupby_app_os_channel', 'unique_count_ip_groupby_app_os_channel', 'unique_count_app_groupby_device_os_channel', 'unique_count_ip_groupby_device_os_channel', 'unique_count_channel_groupby_ip_app_device_os', 'unique_count_os_groupby_ip_app_device_channel', 'unique_count_device_groupby_ip_app_os_channel', 'unique_count_app_groupby_ip_device_os_channel', 'unique_count_ip_groupby_app_device_os_channel']

count_groupby_0_3600 = ['count_groupby_ip_app_os_in_time_range_0_3600_seconds',
 'count_groupby_ip_device_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_device_os_in_time_range_0_3600_seconds',
 'count_groupby_ip_os_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_os_in_time_range_0_3600_seconds',
 'count_groupby_ip_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_device_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_device_os_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_device_os_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_device_in_time_range_0_3600_seconds',
 'count_groupby_ip_device_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_os_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_device_os_channel_in_time_range_0_3600_seconds']

count_groupby_0_21600 = ['count_groupby_ip_os_in_time_range_0_21600_seconds',
 'count_groupby_ip_device_os_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_device_in_time_range_0_21600_seconds',
 'count_groupby_ip_in_time_range_0_21600_seconds',
 'count_groupby_ip_device_os_in_time_range_0_21600_seconds',
 'count_groupby_ip_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_os_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_os_in_time_range_0_21600_seconds',
 'count_groupby_ip_device_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_device_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_device_os_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_os_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_device_os_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_device_in_time_range_0_21600_seconds']

feature_dict = {'raw': raw, 'date_time': date_time, 'time_delta': time_delta, 'unique_count': unique_count, 'count_groupby_0_3600':count_groupby_0_3600, 'count_groupby_0_21600':count_groupby_0_21600}

with open(os.path.join(output_dir,'feature_config.json'), 'w', encoding='utf-8') as f:
    json.dump(feature_dict, f)
