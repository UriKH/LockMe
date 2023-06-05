import shutil
import os

from model.dataset import ModelDataset


def main():
    parent_dir = r'C:\DATA\CelebA'
    output = r'C:\DATA\CelebA\transformed'

    subjects_dict = {}
    with open(os.path.join(parent_dir, r'identity_CelebA.txt')) as ds_table:
        for line in ds_table:
            if line[-1] == '\n':
                line = line[:-1]
            image_name = line.split(' ')[0]
            subject = line.split(' ')[1]
            if subject in subjects_dict.keys():
                subjects_dict[subject].append(image_name)
            else:
                subjects_dict[subject] = [image_name]
        print(f'found: {len(subjects_dict.keys())} subject')

    if not os.path.exists(output):
        os.makedirs(output)
    for subject, images in subjects_dict.items():
        if not os.path.exists(os.path.join(output, f's{subject}')):
            os.makedirs(os.path.join(output, f's{subject}'))
        for image_name in images:
            shutil.copy(os.path.join(parent_dir, r'img_align_celeba\img_align_celeba', image_name),
                        os.path.join(output, f's{subject}', image_name))
        print(f'subject: {subject} finished')
    print('Finished')


if __name__ == '__main__':
    # ModelDataset.create_samples_from_folders(r'C:\DATA\CelebA\subjects', r'C:\DATA\CelebA\transformed')
    # main()

    # transfer data
    # for dir in os.listdir(r'C:\DATA\CelebA\transformed-TRAIN'):
    #     l = len(os.listdir(os.path.join(r'C:\DATA\CelebA\transformed-TRAIN', dir)))
    #     if l <= 10:
    #         print(dir)
    #         shutil.move(os.path.join(r'C:\DATA\CelebA\transformed-TRAIN', dir), r'C:\DATA\CelebA\transformed-TEST')

    test = 0
    for dir in os.listdir(r'C:\DATA\CelebA\transformed-TEST'):
        l = len(os.listdir(os.path.join(r'C:\DATA\CelebA\transformed-TEST', dir)))
        test += l
    print(test)

    train = 0
    for dir in os.listdir(r'C:\DATA\CelebA\transformed-TRAIN'):
        l = len(os.listdir(os.path.join(r'C:\DATA\CelebA\transformed-TRAIN', dir)))
        train += l
    print(train)

    print(f'train test split:\n\t> train:{train/(train + test):.2f}\n\t> test:{test/(train + test):.2f}')