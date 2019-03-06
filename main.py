from data import data_providers
from parser import parser
from models import MODELS

args = parser.parse_args()


if __name__ == '__main__':
    data_provider = data_providers[args.data_id]
    x, y, label = data_provider.get_data()

    get_model = MODELS[args.model_id]
    model = get_model(**data_provider.data_format())

    model.fit(x, y, label)
    model.show_entropy_distribution_animation(data=(x, y, label))
