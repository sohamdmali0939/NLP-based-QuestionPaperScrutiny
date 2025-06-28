def map_to_co(question, return_explanation=False):
    topic_to_co = {
        # CO1: C & Python Programming
        'c programming': 'CO1', 'pointers': 'CO1', 'arrays': 'CO1', 'recursion': 'CO1',
        'functions in c': 'CO1', 'loops in c': 'CO1', 'python programming': 'CO1',
        'lists in python': 'CO1', 'dictionaries': 'CO1',

        # CO2: Data Structures
        'data structures': 'CO2', 'stack': 'CO2', 'queue': 'CO2', 'linked list': 'CO2',
        'binary tree': 'CO2', 'graph': 'CO2', 'dfs': 'CO2', 'bfs': 'CO2',
        'sorting algorithms': 'CO2', 'merge sort': 'CO2', 'quick sort': 'CO2',
        'binary search': 'CO2',

        # CO3: Discrete Mathematics
        'discrete mathematics': 'CO3', 'propositional logic': 'CO3', 'sets': 'CO3',
        'functions': 'CO3', 'relations': 'CO3', 'mathematical logic': 'CO3',
        'linear algebra': 'CO3', 'multivariable calculus': 'CO3',

        # CO4: Operating Systems
        'operating system': 'CO4', 'round robin': 'CO4', 'scheduling': 'CO4',
        'semaphores': 'CO4', 'deadlock': 'CO4', 'paging': 'CO4', 'segmentation': 'CO4',
        'file systems': 'CO4', 'memory management': 'CO4', 'linux programming': 'CO4',

        # CO5: DBMS
        'database': 'CO5', 'sql': 'CO5', 'joins': 'CO5', 'normalization': 'CO5',
        '2nf': 'CO5', '3nf': 'CO5', 'acid properties': 'CO5', 'transactions': 'CO5',
        'er diagram': 'CO5', 'distributed databases': 'CO5',

        # CO6: Computer Networks
        'computer networks': 'CO6', 'tcp/ip': 'CO6', 'udp': 'CO6', 'dhcp': 'CO6',
        'smtp': 'CO6', 'osi model': 'CO6', 'switching': 'CO6', 'networking protocols': 'CO6',

        # CO7: Machine Learning & AI
        'machine learning': 'CO7', 'deep learning': 'CO7', 'supervised learning': 'CO7',
        'unsupervised learning': 'CO7', 'regression': 'CO7', 'classification': 'CO7',
        'artificial intelligence': 'CO7', 'pattern recognition': 'CO7',
        'computer vision': 'CO7', 'data mining': 'CO7', 'bioinformatics': 'CO7',
        'computational biology': 'CO7',

        # CO8: Software Engineering
        'software engineering': 'CO8', 'sdlc': 'CO8', 'agile': 'CO8', 'scrum': 'CO8',
        'software testing': 'CO8', 'unit testing': 'CO8', 'software architecture': 'CO8',
        'object-oriented design': 'CO8',

        # CO9: Compiler Design
        'compiler design': 'CO9', 'lexical analysis': 'CO9', 'parsing': 'CO9',
        'syntax analysis': 'CO9', 'dfa': 'CO9', 'nfa': 'CO9', 'automata': 'CO9',
        'formal languages': 'CO9', 'theory of computation': 'CO9',

        # CO10: Web and Mobile Development
        'web development': 'CO10', 'html': 'CO10', 'css': 'CO10', 'javascript': 'CO10',
        'mobile app development': 'CO10', 'flutter': 'CO10', 'react': 'CO10',
        'human-computer interaction': 'CO10',

        # CO11: Cloud Computing
        'cloud computing': 'CO11', 'cloud infrastructure': 'CO11', 'cloud security': 'CO11',

        # CO12: DevOps
        'devops': 'CO12', 'jenkins': 'CO12', 'docker': 'CO12', 'kubernetes': 'CO12',

        # CO13: Cyber Security
        'cyber security': 'CO13', 'cryptography': 'CO13', 'encryption': 'CO13',
        'network security': 'CO13', 'authentication': 'CO13',

        # CO14: IoT & Embedded Systems
        'iot': 'CO14', 'real-time systems': 'CO14', 'embedded systems': 'CO14',
        'embedded c': 'CO14', 'digital signal processing': 'CO14',

        # CO15: Data Analysis
        'exploratory data analysis': 'CO15', 'eda': 'CO15',

        # CO16: NLP
        'natural language processing': 'CO16', 'nlp': 'CO16', 'tokenization': 'CO16',
        'pos tagging': 'CO16', 'named entity recognition': 'CO16',

        # CO17: Big Data
        'big data': 'CO17', 'hadoop': 'CO17', 'spark': 'CO17',

        # CO18: Digital Logic
        'digital logic': 'CO18', 'logic gates': 'CO18', 'flip flops': 'CO18',
        'sequential circuits': 'CO18',

        # CO19: OOP
        'object-oriented programming': 'CO19', 'inheritance': 'CO19',
        'polymorphism': 'CO19', 'encapsulation': 'CO19',

        # CO20: Parallel & Distributed Computing
        'parallel computing': 'CO20', 'distributed systems': 'CO20',
        'parallel algorithms': 'CO20', 'parallel programming': 'CO20',

        # CO21: Optimization
        'optimization techniques': 'CO21', 'linear programming': 'CO21',
        'advanced algorithms': 'CO21',

        # CO22: AR/VR
        'augmented reality': 'CO22', 'virtual reality': 'CO22',
    }

    question_lower = question.lower()
    for topic, co in topic_to_co.items():
        if topic in question_lower:
            explanation = f"The keyword '{topic}' maps the question to {co}."
            return (co, explanation) if return_explanation else co

    default_explanation = "No relevant topic keyword matched; defaulting to COX."
    return ("COX", default_explanation) if return_explanation else "COX"
