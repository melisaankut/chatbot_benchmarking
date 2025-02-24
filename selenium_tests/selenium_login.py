import time
import config

from selenium.webdriver.common.by import By

def login(driver):
    email_input = driver.find_element(By.XPATH, "//input[@id='email']")
    email_input.send_keys(config.CHATBOT_USERNAME)

    password_input = driver.find_element(By.XPATH, "//input[@type='password']")
    password_input.send_keys(config.CHATBOT_PASSWORD)

    login_button = driver.find_element(By.XPATH, "//button[@type='button' and contains(@class, 'ant-btn-primary')]")
    login_button.click()

    time.sleep(2)
    print("Login successfully.")