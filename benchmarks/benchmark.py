#!/usr/bin/env python3
"""HarnessKit Benchmark Suite — Measuring fuzzy edit recovery rates.

Simulates realistic LLM edit failures and compares exact matching (baseline)
against HarnessKit's fuzzy matching cascade.

Usage:
    python3 benchmarks/benchmark.py
"""

import sys
import os
import textwrap

# Add parent dir so we can import hk
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hk import find_best_match, find_exact_matches, AmbiguousMatchError


# ---------------------------------------------------------------------------
# Benchmark cases: (name, category, file_content, llm_old_text, new_text)
#
# Each case represents a realistic scenario where an LLM produces old_text
# that doesn't exactly match the file. HarnessKit should recover; exact match won't.
# ---------------------------------------------------------------------------

BENCHMARKS = []

def bench(name, category, content, old_text, new_text="REPLACED"):
    """Register a benchmark case."""
    BENCHMARKS.append((name, category, content, old_text, new_text))


# ===================== WHITESPACE MISMATCHES =====================

bench(
    "Tabs vs spaces",
    "whitespace",
    "def process(data):\n\tresult = transform(data)\n\treturn result\n",
    "def process(data):\n    result = transform(data)\n    return result\n",
)

bench(
    "Trailing whitespace added",
    "whitespace",
    "class Config:\n    debug = False\n    verbose = True\n",
    "class Config:  \n    debug = False  \n    verbose = True  \n",
)

bench(
    "Trailing whitespace stripped",
    "whitespace",
    "class Config:  \n    debug = False  \n    verbose = True  \n",
    "class Config:\n    debug = False\n    verbose = True\n",
)

bench(
    "Indentation drift (2→4 spaces)",
    "whitespace",
    "def run():\n  x = 1\n  y = 2\n  return x + y\n",
    "def run():\n    x = 1\n    y = 2\n    return x + y\n",
)

bench(
    "Blank line added by model",
    "whitespace",
    "import os\nimport sys\n\ndef main():\n    pass\n",
    "import os\nimport sys\n\n\ndef main():\n    pass\n",
)

bench(
    "Blank line removed by model",
    "whitespace",
    "import os\nimport sys\n\n\ndef main():\n    pass\n",
    "import os\nimport sys\n\ndef main():\n    pass\n",
)

bench(
    "Mixed tabs and spaces",
    "whitespace",
    "if True:\n\t    x = 1\n\t    y = 2\n",
    "if True:\n        x = 1\n        y = 2\n",
)

bench(
    "Windows line endings (CRLF→LF)",
    "whitespace",
    "line one\nline two\nline three\n",
    "line one\r\nline two\r\nline three\r\n",
)

# ===================== MINOR HALLUCINATIONS =====================

bench(
    "Variable name typo (result→results)",
    "hallucination",
    "def fetch():\n    result = get_data()\n    return result\n",
    "def fetch():\n    results = get_data()\n    return results\n",
)

bench(
    "Comment paraphrased",
    "hallucination",
    "# Process the input data and return transformed output\ndef process(data):\n    return transform(data)\n",
    "# Process input data and return the transformed result\ndef process(data):\n    return transform(data)\n",
)

bench(
    "String quote style (single→double)",
    "hallucination",
    "name = 'hello world'\nprint(name)\n",
    'name = "hello world"\nprint(name)\n',
)

bench(
    "Extra pass statement",
    "hallucination",
    "class MyHandler:\n    def handle(self, req):\n        return self.process(req)\n",
    "class MyHandler:\n    def handle(self, req):\n        pass\n        return self.process(req)\n",
)

bench(
    "Missing type hint",
    "hallucination",
    "def calculate(x: int, y: int) -> int:\n    return x + y\n",
    "def calculate(x, y):\n    return x + y\n",
)

bench(
    "Slightly wrong method name",
    "hallucination",
    "    def get_user_name(self):\n        return self.name\n",
    "    def get_username(self):\n        return self.name\n",
)

bench(
    "Import alias difference",
    "hallucination",
    "import numpy as np\ndata = np.array([1, 2, 3])\n",
    "import numpy as np\ndata = numpy.array([1, 2, 3])\n",
)

# ===================== LINE NUMBER DRIFT =====================

bench(
    "Context shifted by 2 lines (prepended)",
    "line_drift",
    "# header\n# license\nimport os\nimport sys\n\ndef main():\n    print('hello')\n    return 0\n",
    "import os\nimport sys\n\ndef main():\n    print('hello')\n    return 0\n",
)

bench(
    "Surrounding context changed",
    "line_drift",
    "def alpha():\n    pass\n\ndef target_function():\n    x = compute()\n    return x\n\ndef omega():\n    pass\n",
    "def target_function():\n    x = compute()\n    return x\n",
)

bench(
    "Extra decorator added above",
    "line_drift",
    "class API:\n    @auth_required\n    @cache(ttl=60)\n    def get_items(self):\n        return self.db.query()\n",
    "class API:\n    @cache(ttl=60)\n    def get_items(self):\n        return self.db.query()\n",
)

# ===================== PARTIAL MATCHES =====================

bench(
    "Model only reproduces first 3 of 5 lines",
    "partial",
    "def validate(data):\n    if not data:\n        raise ValueError('empty')\n    if not isinstance(data, dict):\n        raise TypeError('not dict')\n",
    "def validate(data):\n    if not data:\n        raise ValueError('empty')\n",
)

bench(
    "Model reproduces middle section",
    "partial",
    "# preamble\nimport os\n\ndef setup():\n    config = load()\n    validate(config)\n    return config\n\n# postamble\n",
    "def setup():\n    config = load()\n    validate(config)\n    return config\n",
)

# ===================== REAL-WORLD STR_REPLACE FAILURES =====================

bench(
    "str_replace: whitespace + hallucination combo",
    "real_world",
    textwrap.dedent("""\
        def handle_request(self, request):
            # Validate the incoming request data
            if not request.is_valid():
                logger.warning("Invalid request received")
                return Response(status=400)
            
            # Process the validated request
            result = self.processor.process(request)
            return Response(data=result, status=200)
    """),
    textwrap.dedent("""\
        def handle_request(self, request):
            # Validate incoming request data
            if not request.is_valid():
                logger.warning("Invalid request received")
                return Response(status=400)

            # Process the validated request
            result = self.processor.process(request)
            return Response(data=result, status=200)
    """),
)

bench(
    "str_replace: model normalizes empty lines",
    "real_world",
    "class Parser:\n\n\n    def __init__(self):\n        self.tokens = []\n\n\n    def parse(self, text):\n        return self._tokenize(text)\n",
    "class Parser:\n\n    def __init__(self):\n        self.tokens = []\n\n    def parse(self, text):\n        return self._tokenize(text)\n",
)

bench(
    "str_replace: indentation level wrong",
    "real_world",
    "    async def fetch_data(self):\n        async with aiohttp.ClientSession() as session:\n            async with session.get(self.url) as resp:\n                return await resp.json()\n",
    "async def fetch_data(self):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(self.url) as resp:\n            return await resp.json()\n",
)

bench(
    "apply_patch: fuzz factor context mismatch",
    "real_world",
    textwrap.dedent("""\
        def calculate_total(items):
            subtotal = sum(item.price for item in items)
            tax = subtotal * 0.08
            discount = get_discount(subtotal)
            total = subtotal + tax - discount
            return round(total, 2)
    """),
    textwrap.dedent("""\
        def calculate_total(items):
            subtotal = sum(i.price for i in items)
            tax = subtotal * 0.08
            discount = get_discount(subtotal)
            total = subtotal + tax - discount
            return round(total, 2)
    """),
)

bench(
    "Model reproduces from memory (multiple small diffs)",
    "real_world",
    textwrap.dedent("""\
        def init_app(config_path):
            config = load_config(config_path)
            db = Database(config['db_url'])
            cache = Redis(config['redis_url'])
            app = Application(db=db, cache=cache)
            app.register_routes()
            return app
    """),
    textwrap.dedent("""\
        def init_app(config_path):
            config = load_config(config_path)
            db = Database(config["db_url"])
            cache = Redis(config["redis_url"])
            app = Application(db=db, cache=cache)
            app.register_routes()
            return app
    """),
)

bench(
    "Docstring formatting difference",
    "real_world",
    textwrap.dedent('''\
        def connect(host, port, timeout=30):
            """Connect to the remote server.
            
            Args:
                host: Server hostname
                port: Server port number
                timeout: Connection timeout in seconds
            """
            sock = socket.create_connection((host, port), timeout)
            return sock
    '''),
    textwrap.dedent('''\
        def connect(host, port, timeout=30):
            """Connect to the remote server.

            Args:
                host (str): Server hostname
                port (int): Server port number
                timeout (int): Connection timeout in seconds
            """
            sock = socket.create_connection((host, port), timeout)
            return sock
    '''),
)


# ===================== MULTI-LANGUAGE REAL FAILURES =====================
# Cases modeled after real GitHub issues from Aider, Cursor, Continue, etc.

bench(
    "TypeScript: optional chaining hallucinated",
    "hallucination",
    "const name = user.profile.name;\nconst email = user.profile.email;\n",
    "const name = user?.profile?.name;\nconst email = user?.profile?.email;\n",
)

bench(
    "Rust: lifetime annotation dropped",
    "hallucination",
    "fn parse<'a>(input: &'a str) -> Result<&'a str, Error> {\n    Ok(&input[1..])\n}\n",
    "fn parse(input: &str) -> Result<&str, Error> {\n    Ok(&input[1..])\n}\n",
)

bench(
    "Go: error handling rewritten",
    "hallucination",
    "result, err := doWork(ctx)\nif err != nil {\n\treturn fmt.Errorf(\"doWork failed: %w\", err)\n}\n",
    "result, err := doWork(ctx)\nif err != nil {\n\treturn fmt.Errorf(\"failed to do work: %w\", err)\n}\n",
)

bench(
    "JSX: self-closing vs explicit close",
    "hallucination",
    '<Button onClick={handleClick} className="primary" />\n',
    '<Button onClick={handleClick} className="primary"></Button>\n',
)

bench(
    "Python: f-string vs .format()",
    "hallucination",
    'msg = f"Hello {name}, you have {count} items"\nprint(msg)\n',
    'msg = "Hello {}, you have {} items".format(name, count)\nprint(msg)\n',
)

bench(
    "SQL: case sensitivity difference",
    "hallucination",
    "SELECT u.id, u.name FROM users u WHERE u.active = TRUE ORDER BY u.name;\n",
    "select u.id, u.name from users u where u.active = true order by u.name;\n",
)

# ===================== COMPLEX WHITESPACE =====================

bench(
    "Python: nested indentation 3-deep wrong",
    "whitespace",
    "class Server:\n    def handle(self, req):\n        if req.valid:\n            for item in req.items:\n                self.process(item)\n",
    "class Server:\n  def handle(self, req):\n    if req.valid:\n      for item in req.items:\n        self.process(item)\n",
)

bench(
    "YAML: indentation matters semantically",
    "whitespace",
    "services:\n  web:\n    image: nginx:latest\n    ports:\n      - \"80:80\"\n    volumes:\n      - ./html:/usr/share/nginx/html\n",
    "services:\n    web:\n        image: nginx:latest\n        ports:\n            - \"80:80\"\n        volumes:\n            - ./html:/usr/share/nginx/html\n",
)

bench(
    "Alignment whitespace in table",
    "whitespace",
    "NAME      = 'myapp'\nVERSION   = '1.0.0'\nAUTHOR    = 'dev'\n",
    "NAME = 'myapp'\nVERSION = '1.0.0'\nAUTHOR = 'dev'\n",
)

# ===================== CONTEXT DRIFT =====================

bench(
    "Model saw old import list",
    "line_drift",
    "import os\nimport sys\nimport json\nimport logging\nfrom pathlib import Path\n\ndef main():\n    logging.info('start')\n",
    "import os\nimport sys\nimport json\nfrom pathlib import Path\n\ndef main():\n    logging.info('start')\n",
)

bench(
    "Function was renamed since model read file",
    "line_drift",
    "def process_request(req):\n    data = validate(req)\n    return transform(data)\n",
    "def handle_request(req):\n    data = validate(req)\n    return transform(data)\n",
)

# ===================== HARD CASES =====================
# These push the boundaries — lower similarity, multiple differences

bench(
    "Multiple hallucinations in one block",
    "hard",
    textwrap.dedent("""\
        async def fetch_users(db, limit=100, offset=0):
            query = "SELECT * FROM users WHERE active = true LIMIT $1 OFFSET $2"
            rows = await db.fetch(query, limit, offset)
            return [User(**row) for row in rows]
    """),
    textwrap.dedent("""\
        async def fetch_users(database, limit=100, offset=0):
            query = "SELECT * FROM users WHERE active = TRUE LIMIT $1 OFFSET $2"
            rows = await database.fetch(query, limit, offset)
            return [User(**r) for r in rows]
    """),
)

bench(
    "Comment + whitespace + hallucination triple",
    "hard",
    textwrap.dedent("""\
        class TokenBucket:
            # Rate limiter using token bucket algorithm
            def __init__(self, rate, capacity):
                self.rate = rate
                self.capacity = capacity
                self.tokens = capacity
                self.last_refill = time.monotonic()
    """),
    textwrap.dedent("""\
        class TokenBucket:
            # Token bucket rate limiter
            def __init__(self, rate, capacity):
                self.rate = rate
                self.capacity = capacity
                self.tokens = capacity
                self.last_refill = time.monotonic()
    """),
)

bench(
    "Large block with scattered small diffs",
    "hard",
    textwrap.dedent("""\
        def setup_logging(config):
            level = getattr(logging, config.get('level', 'INFO'))
            fmt = config.get('format', '%(asctime)s %(levelname)s %(message)s')
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt))
            logger = logging.getLogger('myapp')
            logger.setLevel(level)
            logger.addHandler(handler)
            if config.get('file'):
                fh = logging.FileHandler(config['file'])
                fh.setFormatter(logging.Formatter(fmt))
                logger.addHandler(fh)
            return logger
    """),
    textwrap.dedent("""\
        def setup_logging(config):
            level = getattr(logging, config.get("level", "INFO"))
            fmt = config.get("format", "%(asctime)s %(levelname)s %(message)s")
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt))
            logger = logging.getLogger("myapp")
            logger.setLevel(level)
            logger.addHandler(handler)
            if config.get("file"):
                fh = logging.FileHandler(config["file"])
                fh.setFormatter(logging.Formatter(fmt))
                logger.addHandler(fh)
            return logger
    """),
)

bench(
    "Model compresses multiline to single line",
    "hard",
    "results = [\n    process(x)\n    for x in data\n    if x.valid\n]\n",
    "results = [process(x) for x in data if x.valid]\n",
)

bench(
    "Decorator order swapped + whitespace",
    "hard",
    "    @staticmethod\n    @lru_cache(maxsize=128)\n    def compute(x, y):\n        return expensive_op(x, y)\n",
    "    @lru_cache(maxsize=128)\n    @staticmethod\n    def compute(x, y):\n        return expensive_op(x, y)\n",
)

bench(
    "Model adds semicolons (JS→TS habit)",
    "hallucination",
    "const x = 1\nconst y = 2\nconst z = x + y\nconsole.log(z)\n",
    "const x = 1;\nconst y = 2;\nconst z = x + y;\nconsole.log(z);\n",
)

bench(
    "C: brace style K&R vs Allman",
    "hallucination",
    "int main() {\n    printf(\"hello\\n\");\n    return 0;\n}\n",
    "int main()\n{\n    printf(\"hello\\n\");\n    return 0;\n}\n",
)

bench(
    "Model expands abbreviated variable names",
    "hallucination",
    "for i, v in enumerate(vals):\n    res[i] = fn(v)\n",
    "for index, value in enumerate(vals):\n    res[index] = fn(value)\n",
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmarks():
    exact_pass = 0
    hk_pass = 0
    total = len(BENCHMARKS)
    
    results_by_category = {}
    
    print("=" * 78)
    print("  HarnessKit Benchmark Suite — Fuzzy Edit Recovery")
    print("=" * 78)
    print()
    
    for name, category, content, old_text, new_text in BENCHMARKS:
        # Exact match baseline
        exact_matches = find_exact_matches(content, old_text)
        exact_ok = len(exact_matches) == 1
        
        # HarnessKit fuzzy match
        try:
            match = find_best_match(content, old_text, threshold=0.6)
            hk_ok = match is not None
            match_type = match.match_type if match else "—"
            confidence = f"{match.confidence:.0%}" if match else "—"
        except AmbiguousMatchError:
            hk_ok = False
            match_type = "ambiguous"
            confidence = "—"
        
        if exact_ok:
            exact_pass += 1
        if hk_ok:
            hk_pass += 1
        
        if category not in results_by_category:
            results_by_category[category] = {"exact": 0, "hk": 0, "total": 0}
        results_by_category[category]["total"] += 1
        if exact_ok:
            results_by_category[category]["exact"] += 1
        if hk_ok:
            results_by_category[category]["hk"] += 1
        
        exact_sym = "✅" if exact_ok else "❌"
        hk_sym = "✅" if hk_ok else "❌"
        
        print(f"  {exact_sym} → {hk_sym}  [{match_type:>12s} {confidence:>5s}]  {name}")
    
    print()
    print("=" * 78)
    print("  Results by Category")
    print("=" * 78)
    print()
    print(f"  {'Category':<25s} {'Exact Match':>12s} {'HarnessKit':>12s} {'Recovery':>10s}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 12} {'─' * 10}")
    
    for cat, data in results_by_category.items():
        cat_label = cat.replace("_", " ").title()
        exact_rate = f"{data['exact']}/{data['total']}"
        hk_rate = f"{data['hk']}/{data['total']}"
        if data['total'] - data['exact'] > 0:
            recovery = f"{(data['hk'] - data['exact']) / (data['total'] - data['exact']):.0%}"
        else:
            recovery = "—"
        print(f"  {cat_label:<25s} {exact_rate:>12s} {hk_rate:>12s} {recovery:>10s}")
    
    print(f"  {'─' * 25} {'─' * 12} {'─' * 12} {'─' * 10}")
    exact_rate_total = f"{exact_pass}/{total}"
    hk_rate_total = f"{hk_pass}/{total}"
    if total - exact_pass > 0:
        recovery_total = f"{(hk_pass - exact_pass) / (total - exact_pass):.0%}"
    else:
        recovery_total = "—"
    print(f"  {'TOTAL':<25s} {exact_rate_total:>12s} {hk_rate_total:>12s} {recovery_total:>10s}")
    
    print()
    print(f"  Exact match baseline:  {exact_pass}/{total} ({exact_pass/total:.0%})")
    print(f"  HarnessKit fuzzy:      {hk_pass}/{total} ({hk_pass/total:.0%})")
    if total > exact_pass:
        print(f"  Recovery rate:         {hk_pass - exact_pass}/{total - exact_pass} failed edits recovered ({(hk_pass - exact_pass)/(total - exact_pass):.0%})")
    print()
    
    return exact_pass, hk_pass, total


if __name__ == "__main__":
    exact, hk, total = run_benchmarks()
    sys.exit(0 if hk >= exact else 1)
