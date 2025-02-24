from selenium import webdriver
import config

def get_web_driver():
    options = webdriver.ChromeOptions()
    options.add_argument(config.WINDOW_SIZE)
    driver = webdriver.Chrome(options=options)

    return driver