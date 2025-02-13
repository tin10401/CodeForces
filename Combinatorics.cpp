vi primes, first_divisor(MX), DIV[MX];
bitset<MX> primeBits;
vll mu(MX);

void generatePrime() {  primeBits.set(2);   
    for(int i = 3; i < MX; i += 2) primeBits.set(i);
    for(int i = 2; i * i < MX; i += (i == 2 ? 1 : 2)) {    
        if(primeBits[i]) {  
            for(int j = i; j * i < MX; j += 2) {    primeBits.reset(i * j); }
        }
    }
    for(int i = 2; i < MX; i++) {    
        if(primeBits[i]) {  
            for(int j = i; j < MX; j += i) {    if(first_divisor[j] == 0) first_divisor[j] = i; }
        }
    }
    for(int i = 0; i < MX; i++ ) {  if(primeBits[i]) {  primes.pb(i); } }   

	mu[1] = 1;
    for(int i = 1; i < MX; i++) {   
        if(!primeBits[i]) continue;
        for(int j = i; j < MX; j += i) {   
            if(j >= i * 2) mu[j] -= mu[i];
			DIV[j].pb(i);
        }
    }
}

vi factor(int x) {
    vi a;
    for(int i = 1; i * i <= x; i++) {
        if(x % i == 0) {
            a.pb(i);
            if(i * i != x) a.pb(x / i);
        }
    }
    return a;
}

class Combinatoric {    
    public: 
    int n;  
    vll fact, inv;   
    Combinatoric(int n) {   
        this->n = n;    
        fact.rsz(n + 1), inv.rsz(n + 1);
        init();
    }
        
    void init() {   
        fact[0] = 1;
        for(int i = 1; i <= n; i++) {   
            fact[i] = (fact[i - 1] * i) % MOD;
        }
        inv[n] = modExpo(fact[n], MOD - 2, MOD);
        for(int i = n - 1; i >= 0; i--) {   
            inv[i] = (inv[i + 1] * (i + 1)) % MOD;
        }
    }
    
    ll choose(int a, int b) {  
        if(a < b) return 0;
        return fact[a] * inv[b] % MOD * inv[a - b] % MOD;
    }
	
	ll choose_no_mod(ll a, ll b) {
		ll res = 1;
        for(int i = 0; i < b; i++) res *= (a - i);
        for(int i = 2; i <= b; i++) res /= i;
        return res;
    }
};

ll XOR(ll n) {    
	if(n % 4 == 0) return n;
	if(n % 4 == 1) return 1;    
	if(n % 4 == 2) return n + 1;
	return 0;
};

vll countBit(ll n) {    
	int m = 62;
	vll cnt(m);
    auto f = [&](ll A, ll B) -> ll {
        return A + B;
    };
	while(n > 0) {  
		ll msb = log2(n);
		ll c = (1LL << msb); 
		cnt[msb] = f(cnt[msb], n - c + 1);
		n -= c;
		c >>= 1;
		for(int i = 0; i < msb; i++) {  
			cnt[i] += c; // careful with MOD
		}
	}
	return cnt;
};

ll get_mask(ll a, ll k) { // get bit_mask representation in base k
    ll res = 0;
    int cnt = 0;
    while(a > 0) {
        ll digit = a % k;
        if(digit > 1) return -1;
        if(digit == 1) res |= (1LL << cnt);
        a /= k;
        cnt++;
    }
    return res;
}

vi get_pair_gcd(vi& a) {
    int m = MAX(a);
    vi f(m + 1), g(m + 1);
    for(auto& x : a) f[x]++;
    for(int i = 1; i <= m; i++) {
        for(int j = i; j <= m; j += i) {
            g[i] += f[j];
        }
    }
    for(int i = m; i >= 1; i--) {
        for(int j = i * 2; j <= m; j += i) {
            g[i] -= g[j];
        }
    }
    return g;
}

int find_y(int x, int p, int c) { // find y such that (x * y) % p == c
    auto egcd = [&](auto self, int a, int b) -> tuple<int, int, int> {
        if (!b) return {a, 1, 0};
        auto [g, s, t] = self(self, b, a % b);
        return make_tuple(g, t, s - (a / b) * t);
    };

    auto modInv = [&](int a, int m) -> int {
        auto [d, inv, _] = egcd(egcd, a, m);
        if (d != 1) return -1;
        inv %= m;
        return inv < 0 ? inv + m : inv;
    };

    auto [d, s, t] = egcd(egcd, x, p);
    if (c % d != 0) return -1;
    int x1 = x / d, p1 = p / d, c1 = c / d;
    int inv = modInv(x1, p1);
    if (inv == -1) return -1;
    int y0 = (ll)c1 * inv % p1;
    return y0 < 0 ? y0 + p1 : y0;
}

//    ll curr = 1;
//    while(curr <= j) {
//        ll add = j / curr;
//        ll last = j / add;
//        if(j + add > MX) break;
//        if(dp[j + add] == -1) {
//            dp[j + add] = dp[j] + 1;
//            q.push(j + add);
//        }
//        curr = last + 1;
//    }

//    int n; cin >> n; // compute distinct prefix_or of an array over all permutation
//    const int N = 2 * n;
//    vi mask(N), dp(N);
//    int zero = 0;
//    while(n--) {
//        int x; cin >> x;
//        zero |= x == 0;
//        mask[x] = x;
//    }
//    const int K = 22;
//    for(int bit = 0; bit < K; bit++) {
//        for(int x = 0;x < N; x++) {
//            if((x >> bit) & 1) mask[x] |= mask[x ^ (1 << bit)];
//        }
//    }
//    for(int x = 0; x < N; x++) {
//        for(int bit = 0; bit < K; bit++) {
//            if((x >> bit) & 1) {
//                int v = dp[x ^ (1 << bit)];
//                if((mask[x] >> bit) & 1) v++;
//                dp[x] = max(dp[x], v);
//            }
//        }
//    }
//    int res = MAX(dp) + zero;
//    cout << res << endl;

//                 for(int other = mask; other; other = (other - 1) & mask) // iterate over all submask of mask
