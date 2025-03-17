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

	iota(all(mu), 0); // for placeholder value
    // mu[1] = 1; // for count of occurences
    for(int i = 1; i < MX; i++) {   
        if(!primeBits[i]) continue;
        for(int j = i; j < MX; j += i) {   
            if(j >= i * 2) mu[j] -= mu[i];
			DIV[j].pb(i);
        }
    }
}

// calculating the number of m * n where gcd(m, n) == 1 && m * n == k
// the answer is the number of 2 ^ (#prime divisor of k)

vi factor_prime(int x) {
    vi d;
    for(auto& p : primes) {
        if(p * p > x) break;
        if(x % p) continue;
        d.pb(p);
        while(x % p == 0) x /= p;
    }
    if(x > 1) d.pb(x);
    return d;
}

ll count_coprime(int up, int x) { // count number from [1 to up] where gcd(num, x) == 1
    auto d = factor_prime(x);
    int N = d.size();
    ll ans = 0;
    for (int mask = 0; mask < (1 << N); mask++) {
        int prod = up, sign = 1;
        for (int i = 0; i < N; i++) {
            if (have_bit(mask, i)) {
                prod /= d[i];
                sign *= -1;
            }
        }
        ans += prod * sign;
    }
    return ans;
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
	
    T nCk(int n, int r) { // change to ll if needed
        T ans = 1;
        for(int i = 1 ; i <= r ; i++) {
            ans *= n - i + 1;
            ans /= i ;   
        }
        return ans ;
    }

    T nCk_increasing_sequence(int l, int r, int len) { // given a range of number from l to r, len k, 
                                                       // return the number of ways to choose those element in increasing order
        return nCk(r - l + len, len);
        // x _ _ _ y
        // # of way to choose the _ unknown value
        // len = pos[y] - pos[x] - 1
    }

//    ll nCk_mod_Lucas_Theorem(int n, int r, int mod) {
//        if(r > n) return 0 ;
//        ll res = 1;
//        while(n && r) {
//            res *= nCk(n % mod, r % mod) ;
//            res %= mod ;
//            n /= mod ;
//            r /= mod ; 
//        }
//        return res ;
//    }
//
//    int nCk_lucas(int n, int r, int mod) {
//        vi ans;
//        for(auto& x : DIV[mod]) {
//            ans.pb(nCk_mod_Lucas_Theorem(n, r, x));
//        }
//        ll res = 0;
//        for(int i = 0; i < int(DIV[mod].size()); i++) {
//            int p = DIV[mod][i];
//            ll m = mod / p;
//            ll inv = modExpo(m, p - 2, p);
//            res = (res + ans[i] * m % mod * inv) % mod;
//        }
//        return res;
//    }

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

const int BITS = 30;
 // add is not the same as merge
template<typename T>
struct xor_basis {
    // A list of basis values sorted in decreasing order, where each value has a unique highest bit.
    // We use a static array instead of a vector for better performance.
    T basis[BITS];
    int n = 0;
 
    T min_value(T start) const {
        if (n == BITS) return 0;
        for (int i = 0; i < n; i++)
            start = min(start, start ^ basis[i]);
        return start;
    }
 
    T max_value(T start = 0) const { 
        if (n == BITS) return (T(1) << BITS) - 1;
        for (int i = 0; i < n; i++)
            start = max(start, start ^ basis[i]);
        return start;
    }
 
    bool add(T x) {
        x = min_value(x);
        if (x == 0) return false;
        basis[n++] = x;
        int k = n - 1;
        // Insertion sort.
        while (k > 0 && basis[k] > basis[k - 1]) {
            swap(basis[k], basis[k - 1]);
            k--;
        }
        // Optional: remove the highest bit of x from other basis elements.
        // for (int i = k - 1; i >= 0; i--)
        //     basis[i] = min(basis[i], basis[i] ^ x);
        return true;
    }
 
    void merge(const xor_basis<T> &other) {
        for (int i = 0; i < other.n && n < BITS; i++)
            add(other.basis[i]);
    }
 
    void merge(const xor_basis<T> &a, const xor_basis<T> &b) {
        if (a.n > b.n) {
            *this = a;
            merge(b);
        } else {
            *this = b;
            merge(a);
        }
    }

    bool operator==(const xor_basis<T> &other) const {
        if(n != other.n) return false;
        for (int i = 0; i < n; i++)
            if(basis[i] != other.basis[i])
                return false;
        return true;
    }
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

// Does the inverse of `submask_sums`; returns the input that produces the given output.
template<typename T_out, typename T_in>
void mobius_transform(int n, vt<T_in> &values) { // remember to set dp[mask] = -dp[mask] if(pct(mask) % 2 == 0) later
    assert(int(values.size()) == 1 << n);
 
    for (int i = 0; i < n; i++) {
        for (int base = 0; base < 1 << n; base += 1 << (i + 1)) {
            for (int mask = base; mask < base + (1 << i); mask++) {
                values[mask + (1 << i)] -= values[mask];
            }
        }
    }
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
