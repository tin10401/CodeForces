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

vi factor_prime(int x) {
    vi d;
    while(x > 1) {
        int t = first_divisor[x];
        d.pb(t);
        while(x % t == 0) x /= t;
    }
    return d;
}

vi factor(int x) {
    vi a;
    for(int i = 1; i * i <= x; i++) {
        if(x % i == 0) {
            a.pb(i);
            if(i * i != x) a.pb(x / i);
        }
    }
	srt(a);
    return a;
}

template<class T> 
class Combinatoric {    
    public: 
    int n;  
    vt<T> fact, inv;   
    Combinatoric(int n) {   
        this->n = n;    
        fact.rsz(n + 1), inv.rsz(n + 1);
        init();
    }
        
    void init() {   
        fact[0] = 1;
        for(int i = 1; i <= n; i++) {   
            fact[i] = fact[i - 1] * i;
        }
        inv[n] = fact[n].inv();
        for(int i = n - 1; i >= 0; i--) {   
            inv[i] = inv[i + 1] * (i + 1);
        }
    }
    
    T choose(int a, int b) {  
        if(a < b) return 0;
        return fact[a] * inv[b] * inv[a - b];
    }
	
	ll choose_no_mod(ll a, ll b) {
		ll res = 1;
        for(int i = 0; i < b; i++) res *= (a - i);
        for(int i = 2; i <= b; i++) res /= i;
        return res;
    }

    T catalan(int k) { // # of pair of balanced bracket of length n is catalan(n / 2)
        if(k == 0) return 1;
        return choose(2 * k, k) - choose(2 * k, k - 1);
    }
};

ll XOR(ll n) {    
	if(n % 4 == 0) return n;
	if(n % 4 == 1) return 1;    
	if(n % 4 == 2) return n + 1;
	return 0;
};

ll AND(ll l, ll r) {
    int shift = 0;
    while (l < r) {
        l >>= 1;
        r >>= 1;
        shift++;
    }
    return l << shift;
}

ll OR(ll l, ll r) {
    int shift = 0;
    while (l < r) {
        l >>= 1;
        r >>= 1;
        shift++;
    }
    return (l << shift) | ((1 << shift) - 1);
}

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

//for(auto& p : primes) {
//	for(int c = m / p; c >= 1; c--) f[c] += f[c * p];
//}
//for (int d = m; d >= 1; d--) {
//    dp[d] = f[d] * d;
//    for (auto &p : primes) {
//        if ((ll)d * p > m) break;
//        dp[d] = max(dp[d], dp[d * p] + (f[d] - f[d * p]) * d);
//    }
//}
//same as 
//for(int i = 1; i <= m; i++) {
//	for(int j = i * 2; j <= m; j++) {
//		g[i] += g[j];
//	}
//}
//for(ll i = m; i >= 1; i--) {
//    dp[i] = f[i] * i;
//    for(int j = i * 2; j <= m; j += i) {
//        dp[i] = max(dp[i], dp[j] + (f[i] - f[j]) * i);
//    }
//}



// number of ways to make up the sum of s is 2 ^ (s - 1)

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
