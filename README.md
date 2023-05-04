# Sparkobs
**temporary title, clear lack of imagination**

## What is it?
Sparkobs is a lightweight package to create observation plans for various instruments, but faster than the speed of light (I might be lying about the speed of light part).

It is inspired by [gwemopt](https://github.com/skyportal/gwemopt) and some of [SkyPortal](https://github.com/skyportal/skyportal)'s code.

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

or on a larger localization, for example:
```bash
python -m sparkobs --telescope="config/ztf.toml" --skymap_url="https://gracedb.ligo.org/api/superevents/MS230502c/files/bayestar.fits.gz,1" --level=0.95
```

Also, you can run the integration tests with
```bash
python -m pytest sparkobs/tests.py
```

Let me know what you think, and feel free to contribute!
