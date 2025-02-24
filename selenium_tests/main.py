import time
import config
import selenium_login
import selenium_response_timer
import selenium_config

def main():
    try:
        driver = selenium_config.get_web_driver()
        driver.get(config.CHATBOT_URL)

        selenium_login.login(driver)
        selenium_response_timer.get_response_time(driver)

    except Exception as e:
        print("Error occurred:", e)

    finally:
        time.sleep(config.END_WAITING_TIME)
        driver.quit()

if __name__ == "__main__":
    main()
