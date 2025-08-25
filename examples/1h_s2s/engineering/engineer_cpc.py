import os
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from engineering_utils import concat_cpc, regrid_cpc, check_and_sort_times, split_date_range
from utils import write_dataset

from configs.cpc import get_config


def process_cpc(raw_data_dir, initial_date, final_date, xs, ys, new_extent):
    """Process CPC data for a given date range."""
    times, daytime, raw_cpc_data = concat_cpc(os.path.join(raw_data_dir, str(initial_date.year)), initial_date, final_date)
    lats, lons, train_data = regrid_cpc(raw_cpc_data, xs, ys, new_extent)
    return times, daytime, lats, lons, train_data


def parallel_process_cpc(raw_data_dir, initial_date, final_date, xs, ys, new_extent, chunk_size=7, num_workers=None):
    """Parallelize CPC preprocessing by splitting into chunks of days."""
    date_ranges = split_date_range(initial_date, final_date, chunk_size)
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_date_range = {
            executor.submit(process_cpc, raw_data_dir, start, end, xs, ys, new_extent): (start, end) 
            for start, end in date_ranges
        }

        for future in as_completed(future_to_date_range):
            try:
                times, daytime, lats, lons, train_data = future.result()
                results.append((times, daytime, lats, lons, train_data))
            except Exception as e:
                print(f"Error processing date range {future_to_date_range[future][0]} to {future_to_date_range[future][1]}: {e}")

    # Combine results into single arrays
    lats = results[0][2]
    lons = results[0][3]
    combined_times = np.concatenate([result[0] for result in results])
    combined_daytime = np.concatenate([result[1] for result in results])
    combined_train_data = np.concatenate([result[4] for result in results])

    # Ensure times are sorted and consistent
    times, daytime, train_data = check_and_sort_times(combined_times, combined_daytime, combined_train_data)

    return times, daytime, lats, lons, train_data


def preprocess_cpc_data(raw_data_dir, xs, ys, new_extent, years, months, num_workers):
    """Loop through years and months and preprocess CPC data."""
    preprocessed_data = []
    for year in years:
        for month in months:
            initial_date = datetime(year, month, 1)
            final_date = datetime(year, month, 30 if month == 6 else 31)  # crude month length handling

            print(f"Preprocessing data for {year}-{month:02}:")
            times, daytime, lats, lons, data = parallel_process_cpc(raw_data_dir, initial_date, final_date, xs, ys, new_extent, num_workers=num_workers)
            preprocessed_data.append((year, month, times, daytime, lats, lons, data))
    
    return preprocessed_data


def process_test_data(preprocessed_data, test_data_dir, test_dates):
    """Extract test data from preprocessed dataset based on given test dates."""
    print(f"Test dates received: {test_dates}")
    all_test_times = []
    all_test_daytime = []
    all_test_data = []

    print("Processing test set...")
    for year_str, month_day_list in test_dates.items():
        year = int(year_str)
        for test_month, test_day in month_day_list:
            print(f"  Processing test date: {year}-{test_month:02}-{test_day:02}")  

            for year_pre, month, times, daytime, lats, lons, data in preprocessed_data:  
                if year_pre == year and month == test_month:
                    target_date = datetime(year, test_month, test_day)
                    
                    times_dates_only = np.array([t.date() for t in times])
                    first_idx = np.where(times_dates_only == target_date.date())[0]

                    if first_idx.size == 0:
                        print(f"Date {target_date} not found!") 
                        continue  

                    first_idx = first_idx[0]
                    day_idxs = slice(first_idx, first_idx + 24)
                    
                    all_test_times.append(times[day_idxs])
                    all_test_daytime.append(daytime[day_idxs])  
                    all_test_data.append(data[day_idxs])

    if all_test_times:
        all_test_times = np.concatenate(all_test_times, axis=0)
        all_test_daytime = np.concatenate(all_test_daytime, axis=0)  
        all_test_data = np.concatenate(all_test_data, axis=0)
    else:
        print("No valid test data found.")
        return   

    os.makedirs(test_data_dir, exist_ok=True)
    write_dataset(all_test_times, all_test_daytime, lats, lons, all_test_data, os.path.join(test_data_dir, "cpc_condtday1h.h5"))


def process_train_data(
    preprocessed_data,
    train_data_dir,
    validation_data_dir,
    test_dates,
    validation_years
):
    """Remove test data from training set, split into training + validation, and save."""
    all_train_times = []
    all_train_daytime = []
    all_train_data = []

    print("Processing train set...")
    for year, month, times, daytime, lats, lons, data in preprocessed_data:
        # Remove test dates from training
        if str(year) in test_dates:
            for test_month, test_day in test_dates[str(year)]:
                if test_month == month:
                    first_idx = np.where(times == datetime(year, test_month, test_day))[0][0]
                    day_idxs = slice(first_idx, first_idx + 24)
                    data = np.delete(data, day_idxs, axis=0)
                    times = np.delete(times, day_idxs, axis=0)
                    daytime = np.delete(daytime, day_idxs, axis=0)

        if len(times) > 0:
            all_train_times.append(times)
            all_train_daytime.append(daytime)
            all_train_data.append(data)

    # Merge all into single arrays
    train_times = np.concatenate(all_train_times, axis=0)
    train_daytime = np.concatenate(all_train_daytime, axis=0)
    train_data = np.concatenate(all_train_data, axis=0)
    
    # Split validation set
    for year in validation_years:
        first_datetime = datetime(year, 6, 1)
        validation_idxs = np.where(train_times == first_datetime)[0][0]
        hours_in_the_summer = 24 * (30 + 31 + 31)  # June+July+August
        select_idxs = slice(validation_idxs, validation_idxs + hours_in_the_summer)
        validation_data = train_data[select_idxs]
        validation_times = train_times[select_idxs]
        validation_daytime = train_daytime[select_idxs]
        train_data = np.delete(train_data, np.s_[validation_idxs:validation_idxs + hours_in_the_summer], axis=0)
        train_times = np.delete(train_times, np.s_[validation_idxs:validation_idxs + hours_in_the_summer], axis=0)
        train_daytime = np.delete(train_daytime, np.s_[validation_idxs:validation_idxs + hours_in_the_summer], axis=0)

    # Save to files
    os.makedirs(validation_data_dir, exist_ok=True)
    write_dataset(validation_times, validation_daytime, lats, lons, validation_data, os.path.join(validation_data_dir, "cpc_condtday1h.h5"))

    os.makedirs(train_data_dir, exist_ok=True)
    write_dataset(train_times, train_daytime, lats, lons, train_data, os.path.join(train_data_dir, "cpc_condtday1h.h5"))


def main():
    config = get_config()

    raw_data_dir = config.raw_data_dir
    test_data_dir = config.test_data_dir
    train_data_dir = config.train_data_dir
    validation_data_dir = config.validation_data_dir
    extent = config.extent
    new_extent = config.new_extent
    num_workers = config.num_workers
    years = config.years
    months = config.months
    validation_years = config.validation_years
    test_dates = config.test_dates

    xs = np.arange(extent[0], extent[1], 1000)
    ys = np.arange(extent[2], extent[3], 1000)[::-1]

    # Preprocess data for train and test
    preprocessed_data = preprocess_cpc_data(raw_data_dir, xs, ys, new_extent, years, months, num_workers)
    process_test_data(preprocessed_data, test_data_dir, test_dates)
    process_train_data(preprocessed_data, train_data_dir, validation_data_dir, test_dates, validation_years)
    

if __name__ == '__main__':
    main()
    print("Preprocessing completed")
