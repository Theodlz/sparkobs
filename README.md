# Sparkobs
**temporary title, clear lack of imagination**

## What is it?
Sparkobs is a lightweight package to create observation plans for various instruments. It is inspired by [gwemopt](https://github.com/skyportal/gwemopt) and some of [SkyPortal](https://github.com/skyportal/skyportal)'s code.

## How to install it?

First clone the repository from github
```bash
git clone https://github.com/Theodlz/sparkobs.git
```

**Optional** Create a virtual environment
```bash
python -m venv env
source env/bin/activate
```
(you will need to have [virtualenv](https://virtualenv.pypa.io/en/latest/) installed)

Then go to the directory and install it
```bash
pip install .
```

## How to use it?
This is very much of an alpha. In the long run, it will probably not even be a package, but its methods might be integrated in [gwemopt](https://github.com/skyportal/gwemopt) instead.

For now, it only works for ZTF, which fields and tiles are precomputed and stored in `data/ztf_fields.joblib`.

You can run it on a skymap given its URL, for example:
```bash
python -m sparkobs --skymap_url="https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/2023/bn230430325/quicklook/glg_healpix_all_bn230430325.fit" --telescope="config/ztf.toml"
```

Also, you can run the integration tests with
```bash
python -m pytest sparkobs/tests
```

Everytime you make modifications to the code, you should rerun `pip install .`, and then run the tests to make everything is still working.

