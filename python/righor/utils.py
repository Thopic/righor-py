import re
from pathlib import Path
from righor import _righor


def load_model(species: str, chain: str, identifier=None):
    """ Load the appropriate model, for example:
    load_model("human", "trb", id="emerson") will load the human trb model based 
    on emerson data. 
    load_model("human", "tra") will load the default trb model
    """
    #    model_dir = .absolute().as_posix()
    path_model = (Path(_righor.__file__).parent / Path("righor.data") / Path("models"))
    if not path_model.exists(): # local mode
        path_model = (Path(_righor.__file__).parent.parent.parent / Path("righor.data") / Path("models"))
    if not path_model.exists():
        raise RuntimeError("Error with the installation. Data files not found.")
    model_dir = path_model.absolute().as_posix()

    try:
        # Just try to load both
        model = _righor.vdj.Model.load_model(species,
                                             chain,
                                             model_dir,
                                             identifier)
    except:
        try:
            model = _righor.vj.Model.load_model(species,
                                                chain,
                                                model_dir,
                                                identifier)
        except:
            if identifier is None:
                raise(ValueError(f"Wrong species ({species}) and/or chain ({chain})")) 
            else:
                raise(ValueError(f"Wrong species ({species}) and/or chain ({chain}) and/or id ({id})"))
    return model



def genes_matching(x: str, model):
    """ Map relatively standard gene name to
        the genes used in Igor/Righor.
        In general return a bit more than needed if there's a doubt
        So TRAV1-1*13 will return all TRAV1-1s, but TRAV1-1*1 will return TRAV1-01*01 and
        TRAV1542 will return all TRAV.
        It's far from perfect.
        @ Arguments:
        * x: V or J gene name, form: TYPE##-##*##, or TYPE##-##
        or TYPE[V,J]##, where ## can be interpreted as digits/letters
        and TYPE = "IGH","IGK","IGL" or "TRB"/"TRA"/"TRG"/"TRD"
        * model: righor.vj.Model or righor.vdj.Model object.
        @ Return:
        * list of righor Gene object (x.name to get their names)
        @ Example:
        "IGHV07" -> ["IGHV7-34-1*01", "IGHV7-34-1*02", "IGHV7-4-1*01",
                     "IGHV7-4-1*02", "IGHV7-4-1*03","IGHV7-4-1*04",
                     "IGHV7-4-1*05", "IGHV7-40*01", "IGHV7-81*01"]
        "TRBV07-002*4 -> ["TRBV7-2*04"]
    """

    regex = (r"^(TRB|TRA|IGH|IGK|IGL|TRG|TRD)(?:\w+)?(V|D|J)"
             r"([\w-]+)?(?:/DV\d+)?(?:\*(\d+))?(?:/OR.*)?$")
    g = re.search(regex, x)

    chain = None
    gene_type = None
    gene_id = None
    allele = None

    if g is None:
        raise ValueError("Gene {} does not have a valid name".format(x))
    chain = g.group(1)
    gene_type = g.group(2)
    gene_id = g.group(3)
    allele = None if g.group(4) is None else int(g.group(4))

    if chain is None or gene_type is None:
        raise ValueError("Gene {} does not have a valid name".format(x))

    # check if gene_id contain something of the form
    # ##-## where ## is a digit or ##S##
    gene_id_1 = None
    gene_id_2 = None
    if gene_id is not None:
        g = re.search(r'(\d+)(?:[-S](\d+))?', gene_id)
        if g is not None:
            if g.span()[1] >= 3 and g.group(2) is not None:
                gene_id_1 = int(g.group(1))
                gene_id_2 = int(g.group(2))
            else:
                gene_id_1 = int(g.group(1))
                
    possible_genes = igor_genes(chain, gene_type, model)


    if allele is not None and gene_id_1 is not None and gene_id_2 is not None:
        guess = [a[-1] for a in possible_genes if a[1] == gene_id_1
                 and a[2] == gene_id_2 and a[4] == allele]
        if guess != []:
            return guess
    if gene_id_1 is not None and gene_id_2 is not None:
        guess = [a[-1] for a in possible_genes if a[1] == gene_id_1
                 and a[2] == gene_id_2]
        if guess != []:
            return guess
    if allele is not None and gene_id_1 is not None:
        guess = [a[-1] for a in possible_genes if a[1] == gene_id_1
                 and a[4] == allele]
        if guess != []:
            return guess
    if gene_id_1 is not None:
        guess = [a[-1] for a in possible_genes if a[1] == gene_id_1]
        if guess != []:
            return guess
        
    # if everything else failed return all
    return [a[-1] for a in possible_genes]


    
def igor_genes(chain: str, gene_type: str, model):
    """ Read the model and return all the genes matching the chain and gene_type.
        chain: TR/IG 
        gene_type: V/J 
        It returns the full gene name, plus the gene family, its name, its allele, plus the Gene object
    """
    regex = (r"(\d+)(?:P)?(?:[\-S](\d+)(?:D)?(?:\-(\d+))?)?"
             r"(?:/DV\d+)?(?:-NL1)?(?:\*(\d+))?")  # match all the IGoR gene names

    lst = []

    list_genes = None
    if gene_type == "V":
        list_genes =  model.v_segments
    elif gene_type == "J":
        list_genes = model.j_segments
    else:
        raise ValueError("Gene type {} is not valid".format(gene_type))

    key = chain + gene_type
    for gene_obj in list_genes:
        gene = gene_obj.name
        try:
            lst.append(
                tuple(
                    [gene] + [None if a is None else int(a) for a in
                              re.search(key + regex, gene).groups()]
                    + [gene_obj]
                ))
        except AttributeError:
            raise ValueError(f"{key} does not match. Check if the gene name and the model are compatible (e.g. TRA for a TRB/IGL model)") from None

        
    return lst

