import torch

PRETRAINED_MODEL = "xlm-roberta-base"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ENVIRONMENT_PATH = ".env"

FETCH_TEST_SET_IGNORE_LABELS = ["Low Content", "Not enough info", "Skip", "Mining and Drilling"]

NUMERIC_FEATURES = [
    "nb_imgs", "nb_links_int", "nb_links_ext", "nb_links_tel", "nb_links_email", "nb_input_txt",
    "nb_button", "nb_meta_desc", "nb_meta_keyw", "nb_numerical_strings", "nb_tags",
    "nb_letters", "nb_distinct_hosts_in_urls", "nb_facebook_deep_links",
    "nb_facebook_shallow_links", "nb_linkedin_deep_links", "nb_linkedin_shallow_links",
    "nb_twitter_deep_links", "nb_twitter_shallow_links", "nb_currency_names",
    "nb_distinct_currencies", "distance_title_final_dn", "longest_subsequence_title_final_dn",
    "nb_youtube_deep_links", "nb_youtube_shallow_links", "nb_vimeo_deep_links",
    "nb_vimeo_shallow_links", "fraction_words_title_initial_dn",
    "fraction_words_title_final_dn", "nb_distinct_words_in_title", "distance_title_initial_dn",
    "longest_subsequence_title_initial_dn"
]
NB_NUMERIC_FEATURES = len(NUMERIC_FEATURES)
NB_LINKS_SVD_COMPONENTS = 70
NB_EXTRA_FEATURES = NB_LINKS_SVD_COMPONENTS + NB_NUMERIC_FEATURES
