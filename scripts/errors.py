# -*- coding: utf-8 -*-

# Wencke Liermann
# Universität Potsdam
# Bachelor Computerlinguistik

# 21/07/2020
# Python 3.7.3
# Windows 8
"""User-defined exceptions for the project classes."""

class CatalogError(Exception):
    pass

class ExistingAuthorError(CatalogError):
    def __init__(self, author):
        self.msg = f"An entry for the author '{author}' already exists."

    def __str__(self):
        return self.msg

class NotExistingAuthorError(CatalogError):
    def __init__(self, author):
        self.msg = f"No entry for the '{author}' exists."

    def __str__(self):
        return self.msg

class NotEnoughAuthorsError(CatalogError):
    def __init__(self, catalog):
        self.msg = f"The catalog '{catalog}' is not trained for atleast two authors."

    def __str__(self):
        return self.msg

class MalformedCatalogError(CatalogError):
    def __init__(self, catalog):
        self.msg = (f"Check the catalog '{catalog}', it should contain two "
                     "columns named 'author_name' and 'pretrained_model'.")

    def __str__(self):
        return self.msg