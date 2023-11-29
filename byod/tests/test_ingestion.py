from byod import HFIngestion

def download_wiki():
    prefix = '' # 'allenai/'
    submixes = ['wikitext'] # 'c4'
    allow_patterns=['*'] # 'en/*'
    token='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',

    hf_ingest = HFIngestion('ingest_hf', '/tmp/wiki_1316', token, prefix, 'refs/convert/parquet', submixes)

    hf_ingest.run()



def download_c4():
    prefix = 'allenai/'
    submixes = ['c4']
    allow_patterns=['en/*']
    token='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',

    hf_ingest = HFIngestion('ingest_hf', '/tmp/c4_en/', token, prefix, None, submixes, allow_patterns)

    hf_ingest.run()

#download_wiki()
download_c4()



