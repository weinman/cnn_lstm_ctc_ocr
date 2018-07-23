# CNN-LSTM-CTC-OCR
# Copyright (C) 2017,2018 Jerod Weinman, Abyaya Lamsal, Benjamin Gafford
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# The list (well, string) of valid output characters
# If any example contains a character not found here, 
# you'll get a runtime error when this is encountered.
out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


"""
Dict for constant time string->label conversion
Attribution: https://stackoverflow.com/questions/36459969/python-convert-list-to-dictionary-with-indexes -- from user: Abhijit
Produces a table of character->index mappings according to out_charset
""" 
out_charset_dict = { key: val for val, key in enumerate( out_charset ) }

"""
Dict for constant time label->string conversion
Produces a table of index->string mappings according to out_charset
"""

int_to_string_dict = dict(enumerate(out_charset))

def num_classes():
    """ Returns length/size of out_charset """
    return len( out_charset )


def label_to_string ( labels ):
    string = ''.join( [int_to_string_dict[c] for c in labels] )
    return string
