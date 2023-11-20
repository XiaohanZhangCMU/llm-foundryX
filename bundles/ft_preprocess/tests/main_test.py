from c4_arxiv import main

def test_main():
    taxis = main.get_taxis()
    assert taxis.count() > 5
