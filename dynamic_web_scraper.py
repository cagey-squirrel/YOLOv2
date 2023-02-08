from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from PIL import Image
import os



def download_image_from_url_to_dir(img_url, dir_path, image_index):
    '''
    Downloads image from a given url to dir_path directory.
    Image name is the name of the dir + '_' + image index
    '''
    response = requests.get(img_url)
    if response.status_code:
        image_name = dir_path + '_' + str(image_index) + '.jpg'
        image_path = os.path.join(dir_path, image_name)
        fp = open(image_path, 'wb')
        fp.write(response.content)
        fp.close()


def scrape_images_from_url_to_dir(url, dir_path):
    '''
    Downloades all images from a page at a given url to directory dir_path.
    If dir with given path doesn't exist it is made
    url leads to page with dynamic content loading, so the page needs to be scrolled iteratively for all images to load
    Images are inside a div with id: 'outer_page_{image_index}'
    Image src is stored in img tag with class: 'absimg' in the 'src' attribute
    '''

    # Making a dir if it does not exist
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


    service = webdriver.chrome.service.Service('C:\Windows\chromedriver.exe')
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging']) # Turns off some annoying warnings
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)

    image_index = 0
    while(True):
        image_index += 1
        try:
            outer_page_div = driver.find_element(By.XPATH, f"//div[@id='outer_page_{image_index}']")
            outer_page_div.location_once_scrolled_into_view # Scrolling into the location of image div so the image loads
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, f"page{image_index}"))) # Wait for image to load

            image_tag = outer_page_div.find_element(By.XPATH, f".//img[@class='absimg']")
            image_source = image_tag.get_attribute('src')
            print(image_source)
            download_image_from_url_to_dir(image_source, dir_path, image_index)

        except NoSuchElementException:
            print(f'Page ended at index {image_index}')
            break


def main():
    url = 'https://www.scribd.com/fullscreen/140579892?access_key=key-abqPvSXwqinM1rGh9SsW&allow_share=true&escape=false&show_recommendations=false&view_mode=scroll'
    dir_path = '34_dvanaest_umetnika'
    scrape_images_from_url_to_dir(url, dir_path)


if __name__ == "__main__":
    main()