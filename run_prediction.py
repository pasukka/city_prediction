from city_prediction.city_predictor import CityPredictor


def main():
    file_name = "data.csv"
    cp = CityPredictor(file_name)
    cp()


if __name__ == '__main__':
    main()
