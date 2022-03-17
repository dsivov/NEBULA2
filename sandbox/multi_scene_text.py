import pandas as pd


def process_full_video():
    save_dir_annotation = '/home/migakol/data/small_lsmdc_test/'
    someone_annotation = '/home/migakol/data/small_lsmdc_test/gt/annotations-original.csv'
    # movies = get_dataset_movies()

    gt_data = pd.read_csv(someone_annotation, encoding='unicode_escape', delimiter='\t')
    # gt_data = gt_data['Her mind wanders for a beat.'].tolist()

    movie = '0033_Amadeus'
    movie_folder = '/dataset/lsmdc/avi/'

    # Go over all the frames of the movie and

if __name__ == '__main__':
    print('Started Mutli Scene Element Processing')