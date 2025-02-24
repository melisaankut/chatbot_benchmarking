from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import config

def get_response_time(driver):
    chat_input = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//input[@type='text']")))

    send_button = driver.find_element(By.XPATH, "//button[contains(@class, 'ant-btn-primary')]")

    chat_input.send_keys(config.CHATBOT_MESSAGE)

    WebDriverWait(driver, 10).until(lambda d: send_button.is_enabled())

    start_time = time.time()

    send_button.click()

    print("Message sent, waiting for response...")

    initial_message_count = len(driver.find_elements(By.XPATH, "//div[contains(@class, 'message-container')]//span[contains(@class, 'message-text')]"))

    WebDriverWait(driver, 30).until(
        lambda d: len(d.find_elements(By.XPATH, "//div[contains(@class, 'message-container')]//span[contains(@class, 'message-text')]")) > initial_message_count
    )

    messages = driver.find_elements(By.XPATH, "//div[contains(@class, 'message-container')]//span[contains(@class, 'message-text')]")

    response_text = messages[-1].text
    response_time = time.time() - start_time

    print(f"Response time: {response_time:.2f} seconds")
    print(f"Response of the chatbot: {response_text}")