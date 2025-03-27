# MAT files

The purpose of this document is to be a log of the steps taken to pick apart MAT files in Python and re-assemble the data stored inside.

## Latest Version of MAT files

The latest version of MAT files (>v7.3) can be accessed and treated as a HDF5 file structure. HDF5 structures are structured, hierarchal files that store data in Groups
which can store further Groups and Datasets. Datasets are the collections of information that we're interested in.

### Structure

The MAT files have the following structure
 - #refs#
 - #subsystem#
 - data_array

#### Refs

This Group contains the actual dataset wanted and metadata describing it. The paths are described by a 1 or 2 character code (e.g. /#refs#/b, /#refs#/0b). This Group
contains a mixture of other Groups and Datasets. HDF5 entities can have assigned attributes providing context about the information stored. These attributes
contain metadata about the data including names of variables and definitions of MATLAB specific data. 

```
#refs#/0A
	attributes:
		H5PATH: b'/#refs#/0A'
		MATLAB_class: b'struct'
		MATLAB_fields: [array([b'C', b'u', b's', b't', b'o', b'm', b'P', b'r', b'o', b'p', b's'],
		       dtype='|S1')
		 array([b'V', b'a', b'r', b'i', b'a', b'b', b'l', b'e', b'C', b'u', b's',
		        b't', b'o', b'm', b'P', b'r', b'o', b'p', b's'], dtype='|S1')
		 array([b'v', b'e', b'r', b's', b'i', b'o', b'n', b'S', b'a', b'v', b'e',
		        b'd', b'F', b'r', b'o', b'm'], dtype='|S1')
		 array([b'm', b'i', b'n', b'C', b'o', b'm', b'p', b'a', b't', b'i', b'b',
		        b'l', b'e', b'V', b'e', b'r', b's', b'i', b'o', b'n'], dtype='|S1')
		 array([b'i', b'n', b'c', b'o', b'm', b'p', b'a', b't', b'i', b'b', b'i',
		        b'l', b'i', b't', b'y', b'M', b's', b'g'], dtype='|S1')
		 array([b'a', b'r', b'r', b'a', b'y', b'P', b'r', b'o', b'p', b's'],
		       dtype='|S1')
		 array([b'd', b'a', b't', b'a'], dtype='|S1')
		 array([b'n', b'u', b'm', b'D', b'i', b'm', b's'], dtype='|S1')
		 array([b'u', b's', b'e', b'V', b'a', b'r', b'N', b'a', b'm', b'e', b's',
		        b'O', b'r', b'i', b'g'], dtype='|S1')
		 array([b'u', b's', b'e', b'D', b'i', b'm', b'N', b'a', b'm', b'e', b's',
		        b'O', b'r', b'i', b'g'], dtype='|S1')
		 array([b'd', b'i', b'm', b'N', b'a', b'm', b'e', b's'], dtype='|S1')
		 array([b'd', b'i', b'm', b'N', b'a', b'm', b'e', b's', b'O', b'r', b'i',
		        b'g'], dtype='|S1')
		 array([b'v', b'a', b'r', b'N', b'a', b'm', b'e', b's'], dtype='|S1')
		 array([b'v', b'a', b'r', b'N', b'a', b'm', b'e', b's', b'O', b'r', b'i',
		        b'g'], dtype='|S1')
		 array([b'n', b'u', b'm', b'R', b'o', b'w', b's'], dtype='|S1')
		 array([b'n', b'u', b'm', b'V', b'a', b'r', b's'], dtype='|S1')
		 array([b'v', b'a', b'r', b'D', b'e', b's', b'c', b'r', b'i', b'p', b't',
		        b'i', b'o', b'n', b's'], dtype='|S1')
		 array([b'v', b'a', b'r', b'U', b'n', b'i', b't', b's'], dtype='|S1')
		 array([b'r', b'o', b'w', b'T', b'i', b'm', b'e', b's'], dtype='|S1')
		 array([b'v', b'a', b'r', b'C', b'o', b'n', b't', b'i', b'n', b'u', b'i',
		        b't', b'y'], dtype='|S1')                                        ]
	Value:
		<HDF5 group "/#refs#/0A" (20 members)>
```

As storing strings is inherently a messy problem 
(different character sets, variable length), strings are stored as character arrays of byte codes. Converting them back to string involves converting the codes to 
characters and then joining the result.

```python
''.join([chr(cc) for cc in char_array])
```

The MATLAB_fields are the names of MATLAB specific pieces of metadata. These pieces of metadata are Groups or Datasets stored under the high-level groups

```
#refs#/0A/CustomProps
	attributes:
		H5PATH: b'/#refs#/0A0A'
		MATLAB_class: b'struct'
		MATLAB_empty: 1
	Value:
		<HDF5 dataset "CustomProps": shape (2,), type "<u8">
#refs#/0A/VariableCustomProps
	attributes:
		H5PATH: b'/#refs#/0A0A'
		MATLAB_class: b'struct'
		MATLAB_empty: 1
	Value:
		<HDF5 dataset "VariableCustomProps": shape (2,), type "<u8">
#refs#/0A/arrayProps
	attributes:
		H5PATH: b'/#refs#/0A0A'
		MATLAB_class: b'struct'
		MATLAB_fields: [array([b'D', b'e', b's', b'c', b'r', b'i', b'p', b't', b'i', b'o', b'n'],
		       dtype='|S1')
		 array([b'U', b's', b'e', b'r', b'D', b'a', b't', b'a'], dtype='|S1')
		 array([b'T', b'a', b'b', b'l', b'e', b'C', b'u', b's', b't', b'o', b'm',
		        b'P', b'r', b'o', b'p', b'e', b'r', b't', b'i', b'e', b's'],
		       dtype='|S1')                                                      ]
	Value:
		<HDF5 group "/#refs#/0A/arrayProps" (3 members)>
#refs#/0A/arrayProps/Description
	attributes:
		H5PATH: b'/#refs#/0A0AarrayProps'
		MATLAB_class: b'char'
		MATLAB_empty: 1
	Value:
		<HDF5 dataset "Description": shape (2,), type "<u8">
#refs#/0A/arrayProps/TableCustomProperties
	attributes:
		H5PATH: b'/#refs#/0A0AarrayProps'
		MATLAB_class: b'struct'
		MATLAB_empty: 1
	Value:
		<HDF5 dataset "TableCustomProperties": shape (2,), type "<u8">
#refs#/0A/arrayProps/UserData
	attributes:
		H5PATH: b'/#refs#/0A0AarrayProps'
		MATLAB_class: b'double'
		MATLAB_empty: 1
	Value:
		<HDF5 dataset "UserData": shape (2,), type "<u8">
```

The purpose of some of these fields has been inferred
 - varNames
     + List of variable names as byte arrays
     + Same order the data is stored in
 - arrayProps/Description
     + Description of the array
     + 2 character array
     + Hasn't been found to be populated yet

Arrays of type |O are references to another part of the file. This is a common approach in these files to save redefining the same data repeatedly. These references
can be used like a path to access the target part of the file.

```python
import h5py
with h5py.File(path,'r') as source:
    #### find reference array called ref_array ####
    # the arrays tend to be 2D arrays of a single row so flattening them simplifies iteration
    for rr in ref_array.flatten():
        # resolve reference
        data = source[rr]
```

The user data tends to be stored under a dataset called 'data'. It's an array of references that redirects to the actual datasets elsewhere in the file.

```python
#refs#/0A/data
	attributes:
		H5PATH: b'/#refs#/0A0A'
		MATLAB_class: b'cell'
	Value:
		<HDF5 dataset "data": shape (3, 1), type "|O">
```

The order the references are stored is the same order as the variable names in varNames. This makes it somewhat easier to target the data you want.

```
varNames[0] -> #refs#/0A/data[0]
```

#### Subsystem

Subsystem contains a single dataset called MCOS. This is a collection of references to arrays in #refs#. Below is a log of the references from UC_setitec.

```
    "/#refs#/5": "<HDF5 dataset \"5\": shape (1, 8976), type \"|u1\">",
    "/#refs#/a": "<HDF5 dataset \"a\": shape (2,), type \"<u8\">",
    "/#refs#/6": "<HDF5 dataset \"6\": shape (1, 1), type \"<u2\">",
    "/#refs#/7": "<HDF5 dataset \"7\": shape (1, 1), type \"<f8\">",
    "/#refs#/8": "<HDF5 dataset \"8\": shape (1, 1), type \"<u2\">",
    "/#refs#/9": "<HDF5 group \"/#refs#/9\" (20 members)>",
    "/#refs#/rb": "<HDF5 dataset \"rb\": shape (1, 1), type \"<u2\">",
    "/#refs#/sb": "<HDF5 dataset \"sb\": shape (1, 1), type \"<f8\">",
    "/#refs#/tb": "<HDF5 dataset \"tb\": shape (1, 1), type \"<u2\">",
    "/#refs#/ub": "<HDF5 group \"/#refs#/ub\" (20 members)>",
    "/#refs#/Nb": "<HDF5 dataset \"Nb\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ob": "<HDF5 dataset \"Ob\": shape (1, 1), type \"<f8\">",
    "/#refs#/Pb": "<HDF5 dataset \"Pb\": shape (1, 1), type \"<u2\">",
    "/#refs#/Qb": "<HDF5 group \"/#refs#/Qb\" (20 members)>",
    "/#refs#/0b": "<HDF5 dataset \"0b\": shape (1, 1), type \"<u2\">",
    "/#refs#/ac": "<HDF5 dataset \"ac\": shape (1, 1), type \"<f8\">",
    "/#refs#/bc": "<HDF5 dataset \"bc\": shape (1, 1), type \"<u2\">",
    "/#refs#/cc": "<HDF5 group \"/#refs#/cc\" (20 members)>",
    "/#refs#/vc": "<HDF5 dataset \"vc\": shape (1, 1), type \"<u2\">",
    "/#refs#/wc": "<HDF5 dataset \"wc\": shape (1, 1), type \"<f8\">",
    "/#refs#/xc": "<HDF5 dataset \"xc\": shape (1, 1), type \"<u2\">",
    "/#refs#/yc": "<HDF5 group \"/#refs#/yc\" (20 members)>",
    "/#refs#/Rc": "<HDF5 dataset \"Rc\": shape (1, 1), type \"<u2\">",
    "/#refs#/Sc": "<HDF5 dataset \"Sc\": shape (1, 1), type \"<f8\">",
    "/#refs#/Tc": "<HDF5 dataset \"Tc\": shape (1, 1), type \"<u2\">",
    "/#refs#/Uc": "<HDF5 group \"/#refs#/Uc\" (20 members)>",
    "/#refs#/dd": "<HDF5 dataset \"dd\": shape (1, 1), type \"<u2\">",
    "/#refs#/ed": "<HDF5 dataset \"ed\": shape (1, 1), type \"<f8\">",
    "/#refs#/fd": "<HDF5 dataset \"fd\": shape (1, 1), type \"<u2\">",
    "/#refs#/gd": "<HDF5 group \"/#refs#/gd\" (20 members)>",
    "/#refs#/zd": "<HDF5 dataset \"zd\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ad": "<HDF5 dataset \"Ad\": shape (1, 1), type \"<f8\">",
    "/#refs#/Bd": "<HDF5 dataset \"Bd\": shape (1, 1), type \"<u2\">",
    "/#refs#/Cd": "<HDF5 group \"/#refs#/Cd\" (20 members)>",
    "/#refs#/Vd": "<HDF5 dataset \"Vd\": shape (1, 1), type \"<u2\">",
    "/#refs#/Wd": "<HDF5 dataset \"Wd\": shape (1, 1), type \"<f8\">",
    "/#refs#/Xd": "<HDF5 dataset \"Xd\": shape (1, 1), type \"<u2\">",
    "/#refs#/Yd": "<HDF5 group \"/#refs#/Yd\" (20 members)>",
    "/#refs#/he": "<HDF5 dataset \"he\": shape (1, 1), type \"<u2\">",
    "/#refs#/ie": "<HDF5 dataset \"ie\": shape (1, 1), type \"<f8\">",
    "/#refs#/je": "<HDF5 dataset \"je\": shape (1, 1), type \"<u2\">",
    "/#refs#/ke": "<HDF5 group \"/#refs#/ke\" (20 members)>",
    "/#refs#/De": "<HDF5 dataset \"De\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ee": "<HDF5 dataset \"Ee\": shape (1, 1), type \"<f8\">",
    "/#refs#/Fe": "<HDF5 dataset \"Fe\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ge": "<HDF5 group \"/#refs#/Ge\" (20 members)>",
    "/#refs#/Ze": "<HDF5 dataset \"Ze\": shape (1, 1), type \"<u2\">",
    "/#refs#/1e": "<HDF5 dataset \"1e\": shape (1, 1), type \"<f8\">",
    "/#refs#/2e": "<HDF5 dataset \"2e\": shape (1, 1), type \"<u2\">",
    "/#refs#/3e": "<HDF5 group \"/#refs#/3e\" (20 members)>",
    "/#refs#/lf": "<HDF5 dataset \"lf\": shape (1, 1), type \"<u2\">",
    "/#refs#/mf": "<HDF5 dataset \"mf\": shape (1, 1), type \"<f8\">",
    "/#refs#/nf": "<HDF5 dataset \"nf\": shape (1, 1), type \"<u2\">",
    "/#refs#/of": "<HDF5 group \"/#refs#/of\" (20 members)>",
    "/#refs#/Hf": "<HDF5 dataset \"Hf\": shape (1, 1), type \"<u2\">",
    "/#refs#/If": "<HDF5 dataset \"If\": shape (1, 1), type \"<f8\">",
    "/#refs#/Jf": "<HDF5 dataset \"Jf\": shape (1, 1), type \"<u2\">",
    "/#refs#/Kf": "<HDF5 group \"/#refs#/Kf\" (20 members)>",
    "/#refs#/4f": "<HDF5 dataset \"4f\": shape (1, 1), type \"<u2\">",
    "/#refs#/5f": "<HDF5 dataset \"5f\": shape (1, 1), type \"<f8\">",
    "/#refs#/6f": "<HDF5 dataset \"6f\": shape (1, 1), type \"<u2\">",
    "/#refs#/7f": "<HDF5 group \"/#refs#/7f\" (20 members)>",
    "/#refs#/pg": "<HDF5 dataset \"pg\": shape (1, 1), type \"<u2\">",
    "/#refs#/qg": "<HDF5 dataset \"qg\": shape (1, 1), type \"<f8\">",
    "/#refs#/rg": "<HDF5 dataset \"rg\": shape (1, 1), type \"<u2\">",
    "/#refs#/sg": "<HDF5 group \"/#refs#/sg\" (20 members)>",
    "/#refs#/Lg": "<HDF5 dataset \"Lg\": shape (1, 1), type \"<u2\">",
    "/#refs#/Mg": "<HDF5 dataset \"Mg\": shape (1, 1), type \"<f8\">",
    "/#refs#/Ng": "<HDF5 dataset \"Ng\": shape (1, 1), type \"<u2\">",
    "/#refs#/Og": "<HDF5 group \"/#refs#/Og\" (20 members)>",
    "/#refs#/8g": "<HDF5 dataset \"8g\": shape (1, 1), type \"<u2\">",
    "/#refs#/9g": "<HDF5 dataset \"9g\": shape (1, 1), type \"<f8\">",
    "/#refs#/0g": "<HDF5 dataset \"0g\": shape (1, 1), type \"<u2\">",
    "/#refs#/ah": "<HDF5 group \"/#refs#/ah\" (20 members)>",
    "/#refs#/th": "<HDF5 dataset \"th\": shape (1, 1), type \"<u2\">",
    "/#refs#/uh": "<HDF5 dataset \"uh\": shape (1, 1), type \"<f8\">",
    "/#refs#/vh": "<HDF5 dataset \"vh\": shape (1, 1), type \"<u2\">",
    "/#refs#/wh": "<HDF5 group \"/#refs#/wh\" (20 members)>",
    "/#refs#/Ph": "<HDF5 dataset \"Ph\": shape (1, 1), type \"<u2\">",
    "/#refs#/Qh": "<HDF5 dataset \"Qh\": shape (1, 1), type \"<f8\">",
    "/#refs#/Rh": "<HDF5 dataset \"Rh\": shape (1, 1), type \"<u2\">",
    "/#refs#/Sh": "<HDF5 group \"/#refs#/Sh\" (20 members)>",
    "/#refs#/bi": "<HDF5 dataset \"bi\": shape (1, 1), type \"<u2\">",
    "/#refs#/ci": "<HDF5 dataset \"ci\": shape (1, 1), type \"<f8\">",
    "/#refs#/di": "<HDF5 dataset \"di\": shape (1, 1), type \"<u2\">",
    "/#refs#/ei": "<HDF5 group \"/#refs#/ei\" (20 members)>",
    "/#refs#/xi": "<HDF5 dataset \"xi\": shape (1, 1), type \"<u2\">",
    "/#refs#/yi": "<HDF5 dataset \"yi\": shape (1, 1), type \"<f8\">",
    "/#refs#/zi": "<HDF5 dataset \"zi\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ai": "<HDF5 group \"/#refs#/Ai\" (20 members)>",
    "/#refs#/Ti": "<HDF5 dataset \"Ti\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ui": "<HDF5 dataset \"Ui\": shape (1, 1), type \"<f8\">",
    "/#refs#/Vi": "<HDF5 dataset \"Vi\": shape (1, 1), type \"<u2\">",
    "/#refs#/Wi": "<HDF5 group \"/#refs#/Wi\" (20 members)>",
    "/#refs#/fj": "<HDF5 dataset \"fj\": shape (1, 1), type \"<u2\">",
    "/#refs#/gj": "<HDF5 dataset \"gj\": shape (1, 1), type \"<f8\">",
    "/#refs#/hj": "<HDF5 dataset \"hj\": shape (1, 1), type \"<u2\">",
    "/#refs#/ij": "<HDF5 group \"/#refs#/ij\" (20 members)>",
    "/#refs#/Bj": "<HDF5 dataset \"Bj\": shape (1, 1), type \"<u2\">",
    "/#refs#/Cj": "<HDF5 dataset \"Cj\": shape (1, 1), type \"<f8\">",
    "/#refs#/Dj": "<HDF5 dataset \"Dj\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ej": "<HDF5 group \"/#refs#/Ej\" (20 members)>",
    "/#refs#/Xj": "<HDF5 dataset \"Xj\": shape (1, 1), type \"<u2\">",
    "/#refs#/Yj": "<HDF5 dataset \"Yj\": shape (1, 1), type \"<f8\">",
    "/#refs#/Zj": "<HDF5 dataset \"Zj\": shape (1, 1), type \"<u2\">",
    "/#refs#/1j": "<HDF5 group \"/#refs#/1j\" (20 members)>",
    "/#refs#/jk": "<HDF5 dataset \"jk\": shape (1, 1), type \"<u2\">",
    "/#refs#/kk": "<HDF5 dataset \"kk\": shape (1, 1), type \"<f8\">",
    "/#refs#/lk": "<HDF5 dataset \"lk\": shape (1, 1), type \"<u2\">",
    "/#refs#/mk": "<HDF5 group \"/#refs#/mk\" (20 members)>",
    "/#refs#/Fk": "<HDF5 dataset \"Fk\": shape (1, 1), type \"<u2\">",
    "/#refs#/Gk": "<HDF5 dataset \"Gk\": shape (1, 1), type \"<f8\">",
    "/#refs#/Hk": "<HDF5 dataset \"Hk\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ik": "<HDF5 group \"/#refs#/Ik\" (20 members)>",
    "/#refs#/2k": "<HDF5 dataset \"2k\": shape (1, 1), type \"<u2\">",
    "/#refs#/3k": "<HDF5 dataset \"3k\": shape (1, 1), type \"<f8\">",
    "/#refs#/4k": "<HDF5 dataset \"4k\": shape (1, 1), type \"<u2\">",
    "/#refs#/5k": "<HDF5 group \"/#refs#/5k\" (20 members)>",
    "/#refs#/nl": "<HDF5 dataset \"nl\": shape (1, 1), type \"<u2\">",
    "/#refs#/ol": "<HDF5 dataset \"ol\": shape (1, 1), type \"<f8\">",
    "/#refs#/pl": "<HDF5 dataset \"pl\": shape (1, 1), type \"<u2\">",
    "/#refs#/ql": "<HDF5 group \"/#refs#/ql\" (20 members)>",
    "/#refs#/Jl": "<HDF5 dataset \"Jl\": shape (1, 1), type \"<u2\">",
    "/#refs#/Kl": "<HDF5 dataset \"Kl\": shape (1, 1), type \"<f8\">",
    "/#refs#/Ll": "<HDF5 dataset \"Ll\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ml": "<HDF5 group \"/#refs#/Ml\" (20 members)>",
    "/#refs#/6l": "<HDF5 dataset \"6l\": shape (1, 1), type \"<u2\">",
    "/#refs#/7l": "<HDF5 dataset \"7l\": shape (1, 1), type \"<f8\">",
    "/#refs#/8l": "<HDF5 dataset \"8l\": shape (1, 1), type \"<u2\">",
    "/#refs#/9l": "<HDF5 group \"/#refs#/9l\" (20 members)>",
    "/#refs#/rm": "<HDF5 dataset \"rm\": shape (1, 1), type \"<u2\">",
    "/#refs#/sm": "<HDF5 dataset \"sm\": shape (1, 1), type \"<f8\">",
    "/#refs#/tm": "<HDF5 dataset \"tm\": shape (1, 1), type \"<u2\">",
    "/#refs#/um": "<HDF5 group \"/#refs#/um\" (20 members)>",
    "/#refs#/Nm": "<HDF5 dataset \"Nm\": shape (1, 1), type \"<u2\">",
    "/#refs#/Om": "<HDF5 dataset \"Om\": shape (1, 1), type \"<f8\">",
    "/#refs#/Pm": "<HDF5 dataset \"Pm\": shape (1, 1), type \"<u2\">",
    "/#refs#/Qm": "<HDF5 group \"/#refs#/Qm\" (20 members)>",
    "/#refs#/0m": "<HDF5 dataset \"0m\": shape (1, 1), type \"<u2\">",
    "/#refs#/an": "<HDF5 dataset \"an\": shape (1, 1), type \"<f8\">",
    "/#refs#/bn": "<HDF5 dataset \"bn\": shape (1, 1), type \"<u2\">",
    "/#refs#/cn": "<HDF5 group \"/#refs#/cn\" (20 members)>",
    "/#refs#/vn": "<HDF5 dataset \"vn\": shape (1, 1), type \"<u2\">",
    "/#refs#/wn": "<HDF5 dataset \"wn\": shape (1, 1), type \"<f8\">",
    "/#refs#/xn": "<HDF5 dataset \"xn\": shape (1, 1), type \"<u2\">",
    "/#refs#/yn": "<HDF5 group \"/#refs#/yn\" (20 members)>",
    "/#refs#/Rn": "<HDF5 dataset \"Rn\": shape (1, 1), type \"<u2\">",
    "/#refs#/Sn": "<HDF5 dataset \"Sn\": shape (1, 1), type \"<f8\">",
    "/#refs#/Tn": "<HDF5 dataset \"Tn\": shape (1, 1), type \"<u2\">",
    "/#refs#/Un": "<HDF5 group \"/#refs#/Un\" (20 members)>",
    "/#refs#/do": "<HDF5 dataset \"do\": shape (1, 1), type \"<u2\">",
    "/#refs#/eo": "<HDF5 dataset \"eo\": shape (1, 1), type \"<f8\">",
    "/#refs#/fo": "<HDF5 dataset \"fo\": shape (1, 1), type \"<u2\">",
    "/#refs#/go": "<HDF5 group \"/#refs#/go\" (20 members)>",
    "/#refs#/zo": "<HDF5 dataset \"zo\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ao": "<HDF5 dataset \"Ao\": shape (1, 1), type \"<f8\">",
    "/#refs#/Bo": "<HDF5 dataset \"Bo\": shape (1, 1), type \"<u2\">",
    "/#refs#/Co": "<HDF5 group \"/#refs#/Co\" (20 members)>",
    "/#refs#/Vo": "<HDF5 dataset \"Vo\": shape (1, 1), type \"<u2\">",
    "/#refs#/Wo": "<HDF5 dataset \"Wo\": shape (1, 1), type \"<f8\">",
    "/#refs#/Xo": "<HDF5 dataset \"Xo\": shape (1, 1), type \"<u2\">",
    "/#refs#/Yo": "<HDF5 group \"/#refs#/Yo\" (20 members)>",
    "/#refs#/hp": "<HDF5 dataset \"hp\": shape (1, 1), type \"<u2\">",
    "/#refs#/ip": "<HDF5 dataset \"ip\": shape (1, 1), type \"<f8\">",
    "/#refs#/jp": "<HDF5 dataset \"jp\": shape (1, 1), type \"<u2\">",
    "/#refs#/kp": "<HDF5 group \"/#refs#/kp\" (20 members)>",
    "/#refs#/Dp": "<HDF5 dataset \"Dp\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ep": "<HDF5 dataset \"Ep\": shape (1, 1), type \"<f8\">",
    "/#refs#/Fp": "<HDF5 dataset \"Fp\": shape (1, 1), type \"<u2\">",
    "/#refs#/Gp": "<HDF5 group \"/#refs#/Gp\" (20 members)>",
    "/#refs#/Zp": "<HDF5 dataset \"Zp\": shape (1, 1), type \"<u2\">",
    "/#refs#/1p": "<HDF5 dataset \"1p\": shape (1, 1), type \"<f8\">",
    "/#refs#/2p": "<HDF5 dataset \"2p\": shape (1, 1), type \"<u2\">",
    "/#refs#/3p": "<HDF5 group \"/#refs#/3p\" (20 members)>",
    "/#refs#/lq": "<HDF5 dataset \"lq\": shape (1, 1), type \"<u2\">",
    "/#refs#/mq": "<HDF5 dataset \"mq\": shape (1, 1), type \"<f8\">",
    "/#refs#/nq": "<HDF5 dataset \"nq\": shape (1, 1), type \"<u2\">",
    "/#refs#/oq": "<HDF5 group \"/#refs#/oq\" (20 members)>",
    "/#refs#/Hq": "<HDF5 dataset \"Hq\": shape (1, 1), type \"<u2\">",
    "/#refs#/Iq": "<HDF5 dataset \"Iq\": shape (1, 1), type \"<f8\">",
    "/#refs#/Jq": "<HDF5 dataset \"Jq\": shape (1, 1), type \"<u2\">",
    "/#refs#/Kq": "<HDF5 group \"/#refs#/Kq\" (20 members)>",
    "/#refs#/4q": "<HDF5 dataset \"4q\": shape (1, 1), type \"<u2\">",
    "/#refs#/5q": "<HDF5 dataset \"5q\": shape (1, 1), type \"<f8\">",
    "/#refs#/6q": "<HDF5 dataset \"6q\": shape (1, 1), type \"<u2\">",
    "/#refs#/7q": "<HDF5 group \"/#refs#/7q\" (20 members)>",
    "/#refs#/pr": "<HDF5 dataset \"pr\": shape (1, 1), type \"<u2\">",
    "/#refs#/qr": "<HDF5 dataset \"qr\": shape (1, 1), type \"<f8\">",
    "/#refs#/rr": "<HDF5 dataset \"rr\": shape (1, 1), type \"<u2\">",
    "/#refs#/sr": "<HDF5 group \"/#refs#/sr\" (20 members)>",
    "/#refs#/Lr": "<HDF5 dataset \"Lr\": shape (1, 1), type \"<u2\">",
    "/#refs#/Mr": "<HDF5 dataset \"Mr\": shape (1, 1), type \"<f8\">",
    "/#refs#/Nr": "<HDF5 dataset \"Nr\": shape (1, 1), type \"<u2\">",
    "/#refs#/Or": "<HDF5 group \"/#refs#/Or\" (20 members)>",
    "/#refs#/8r": "<HDF5 dataset \"8r\": shape (1, 1), type \"<u2\">",
    "/#refs#/9r": "<HDF5 dataset \"9r\": shape (1, 1), type \"<f8\">",
    "/#refs#/0r": "<HDF5 dataset \"0r\": shape (1, 1), type \"<u2\">",
    "/#refs#/as": "<HDF5 group \"/#refs#/as\" (20 members)>",
    "/#refs#/ts": "<HDF5 dataset \"ts\": shape (1, 1), type \"<u2\">",
    "/#refs#/us": "<HDF5 dataset \"us\": shape (1, 1), type \"<f8\">",
    "/#refs#/vs": "<HDF5 dataset \"vs\": shape (1, 1), type \"<u2\">",
    "/#refs#/ws": "<HDF5 group \"/#refs#/ws\" (20 members)>",
    "/#refs#/Ps": "<HDF5 dataset \"Ps\": shape (1, 1), type \"<u2\">",
    "/#refs#/Qs": "<HDF5 dataset \"Qs\": shape (1, 1), type \"<f8\">",
    "/#refs#/Rs": "<HDF5 dataset \"Rs\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ss": "<HDF5 group \"/#refs#/Ss\" (20 members)>",
    "/#refs#/bt": "<HDF5 dataset \"bt\": shape (1, 1), type \"<u2\">",
    "/#refs#/ct": "<HDF5 dataset \"ct\": shape (1, 1), type \"<f8\">",
    "/#refs#/dt": "<HDF5 dataset \"dt\": shape (1, 1), type \"<u2\">",
    "/#refs#/et": "<HDF5 group \"/#refs#/et\" (20 members)>",
    "/#refs#/xt": "<HDF5 dataset \"xt\": shape (1, 1), type \"<u2\">",
    "/#refs#/yt": "<HDF5 dataset \"yt\": shape (1, 1), type \"<f8\">",
    "/#refs#/zt": "<HDF5 dataset \"zt\": shape (1, 1), type \"<u2\">",
    "/#refs#/At": "<HDF5 group \"/#refs#/At\" (20 members)>",
    "/#refs#/Tt": "<HDF5 dataset \"Tt\": shape (1, 1), type \"<u2\">",
    "/#refs#/Ut": "<HDF5 dataset \"Ut\": shape (1, 1), type \"<f8\">",
    "/#refs#/Vt": "<HDF5 dataset \"Vt\": shape (1, 1), type \"<u2\">",
    "/#refs#/Wt": "<HDF5 group \"/#refs#/Wt\" (20 members)>",
    "/#refs#/fu": "<HDF5 dataset \"fu\": shape (1, 1), type \"<u2\">",
    "/#refs#/gu": "<HDF5 dataset \"gu\": shape (1, 1), type \"<f8\">",
    "/#refs#/hu": "<HDF5 dataset \"hu\": shape (1, 1), type \"<u2\">",
    "/#refs#/iu": "<HDF5 group \"/#refs#/iu\" (20 members)>",
    "/#refs#/Bu": "<HDF5 dataset \"Bu\": shape (1, 3), type \"<i4\">",
    "/#refs#/Cu": "<HDF5 dataset \"Cu\": shape (1, 3), type \"|O\">"
```
In the above example one array, #refs#/5, stands out. It is an uint 8 array of 8976 values and is present in the Setitec arrays and DAQ arrays.
    

#### data_array

The 'data_array' dataset is at least a 2D array of size (5,11) that redirects to datasets in #refs#. This represents how the data is stored inside MAT
files. Any dataset regardless of size is broken down into a total of 55 chunks. From testing and plotting, the data is broken down by a process similar to downsampling.

![image](https://user-images.githubusercontent.com/46482002/154973829-a03d1d74-db15-49dc-b963-5cb3a514b6c4.png)

So the data at every other index starting at 0 is stored in chunk zero, starting at 1 stored in chunk 1 etc. Depending on the total size of the stored array, the chunks
may not be all the same size. For example, in UB_setitec.mat the chunks are size 695 and 696 elements.

However, if we resolve the references it doesn't redirect to the chunks.

```python
>>> with h5py.File(path,'r') as source:
	data = source['data_array']
	for ref in data[()].flatten():
		print(source[ref].name,source[ref].shape,source[ref].dtype)
		print(source[ref][()])

		
/#refs#/b (1, 6) uint32
[[3707764736          2          1          1          1          2]]
/#refs#/c (1, 6) uint32
[[3707764736          2          1          1          4          2]]
/#refs#/d (1, 6) uint32
[[3707764736          2          1          1          7          2]]
/#refs#/e (1, 6) uint32
[[3707764736          2          1          1         10          2]]
/#refs#/f (1, 6) uint32
[[3707764736          2          1          1         13          2]]
/#refs#/g (1, 6) uint32
[[3707764736          2          1          1         16          2]]
/#refs#/h (1, 6) uint32
[[3707764736          2          1          1         19          2]]
/#refs#/i (1, 6) uint32
[[3707764736          2          1          1         22          2]]
/#refs#/j (1, 6) uint32
[[3707764736          2          1          1         25          2]]
/#refs#/k (1, 6) uint32
[[3707764736          2          1          1         28          2]]
/#refs#/l (1, 6) uint32
[[3707764736          2          1          1         31          2]]
/#refs#/m (1, 6) uint32
[[3707764736          2          1          1         34          2]]
/#refs#/n (1, 6) uint32
[[3707764736          2          1          1         37          2]]
/#refs#/o (1, 6) uint32
[[3707764736          2          1          1         40          2]]
/#refs#/p (1, 6) uint32
[[3707764736          2          1          1         43          2]]
/#refs#/q (1, 6) uint32
[[3707764736          2          1          1         46          2]]
/#refs#/r (1, 6) uint32
[[3707764736          2          1          1         49          2]]
/#refs#/s (1, 6) uint32
[[3707764736          2          1          1         52          2]]
...
```

The references are in fact the [Timetable](https://uk.mathworks.com/help/matlab/timetables.html) where each entry is the datatime stamp for when the chunks were written to
to the file. The timestamps are 6 element arrays of uint32 values organised in descending order of time increments. The datetime stamps can represent any time whether it be
(hours, mins, secs) or days within a month. As it's a 6 element array and the known duration of the recordings tends to be around an hour, the data likely follows
the following format.

```
Year, Month, Day, Hour, Minute, Second
```

The MATLAB page on representing dates and times can be found [here](https://uk.mathworks.com/help/matlab/matlab_prog/represent-date-and-times-in-MATLAB.html).

These stamps can be used to infer the chronological order the chunks were written. They don't have to be decoded to their original form, instead they can be sorted in
ascending order of values. This example only sorts the 2nd to last column for simplicity. When sorted, the paths increase according to the alphabet. Increments
lower case letters, then upper case and then numbers.

```
#refs#/b
#refs#/c
#refs#/d
#refs#/e
#refs#/f
#refs#/g
#refs#/h
#refs#/i
#refs#/j
#refs#/k
#refs#/l
#refs#/m
#refs#/n
#refs#/o
#refs#/p
#refs#/q
#refs#/r
#refs#/s
#refs#/t
#refs#/u
#refs#/v
#refs#/w
#refs#/x
#refs#/y
#refs#/z
#refs#/A
#refs#/B
#refs#/C
#refs#/D
#refs#/E
#refs#/F
#refs#/G
#refs#/H
#refs#/I
#refs#/J
#refs#/K
#refs#/L
#refs#/M
#refs#/N
#refs#/O
#refs#/P
#refs#/Q
#refs#/R
#refs#/S
#refs#/T
#refs#/U
#refs#/V
#refs#/W
#refs#/X
#refs#/Y
#refs#/Z
#refs#/1
#refs#/2
#refs#/3
#refs#/4
````

If the chunk array is higher dimensional (e.g. 2,5,11) then it multiple "pages" of variables stored within it. So one page can have one set of variables and another can have a different set.

## Extracting Data

One approach to extracting data is searching for groups that contain a member called 'data'. This can be achieved by using the h5py visititems method.

```python
# dictionary to hold items
>> data = {}
# path is the filepath of the target data file #
>> with h5py.File(path,'r') as source:
       def find_data(name,item):
           if isinstance(item,h5py.Group):
	       if 'data' in item:
	           # datasets tend to be (1, x) in size
		   # flattening them can make it easier to plot
	           data_refs = item['data'][()].flatten()
		   data[name] = [source[ref][()] for ref in data_refs]
       source.visititems(find_data)
>> len(data)
55
>> for kk,vv in data.items():
       print(kk, len(vv))
       for val in vv:
           print("\t->\t",val.shape,val.dtype)


#refs#/1j 8
	->	 (1, 695) float64
	->	 (1, 695) float64
	->	 (1, 695) float64
	->	 (1, 695) float64
	->	 (1, 695) float64
	->	 (1, 695) float64
	->	 (1, 695) float64
	->	 (1, 695) float64
#refs#/3e 8
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
#refs#/3p 8
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
	->	 (1, 696) float64
etc.
```

The 'data' member is an array of references that each redirects to a variable. The function *find_data* creates a dictionary of paths to where 'data' was found and stores a list of the arrays found when references are resolved. Each array is the data for a variable.

Data paths are created on a rolling alphabet. It starts off on lowercase (a-z), then uppercase (A-Z) and then numbers (0-9). When it reaches the end it loops back around. This occurs when the datasets have two characters as their name.
 
e.g.
	/#refs#/8f 	->	 (1, 695) float64
	/#refs#/9f 	->	 (1, 695) float64
	/#refs#/0f 	->	 (1, 695) float64
	/#refs#/ag 	->	 (1, 695) float64
	/#refs#/bg 	->	 (1, 695) float64
	/#refs#/cg 	->	 (1, 695) float64
	/#refs#/dg 	->	 (1, 695) float64
	/#refs#/eg 	->	 (1, 695) float64

## Conclusion

The shape of the data_array can be used to infer the number of groups of measurements (rows), holes (columns) and coupons (depth). The groups of measurements are commonly referred to as pages in the program documentation, like pages on a book.

As the relationship between the timestamp in the cell_array and datasets is not known, the pages are instead inferred from the number of variables in each datasets. the function [getMatData](src/dataparser.py#L635) sorts the datasets into groups based on the number of unique variables. For e.g. all the datasets that have 8 variables are placed in one group and those with 3 variables are placed in another. These are used to form the pages in the generated structure. This also makes forming the indexing a lot easier as the measurements are a similar size so there's as few NaN values as possible.

**NOTE: Due to the overhead of Pandas and the size of the data file, forming the DataFrame eats up a lot of system memory and tends to make the host system stall.**

Possible solutions might be reducing the accuracy from float64 to float16 to minimize the amount of memory needed or perhaps avoiding complete dictionary comprehension altogether and building the dataframe up line-by-line.

The generated DataFrame is indexed by a MultiIndex object. It first uses the hole number and coupon number and then by data index. In the cases where the measurement size is different, the indicies that can't be populated have NaN.
