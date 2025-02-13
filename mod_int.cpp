template <int MOD>
struct mod_int {
    int value;
    
    mod_int(long long v = 0) {
        value = int(v % MOD);
        if (value < 0) value += MOD;
    }
    
    mod_int& operator+=(const mod_int &other) {
        value += other.value;
        if (value >= MOD) value -= MOD;
        return *this;
    }
    
    mod_int& operator-=(const mod_int &other) {
        value -= other.value;
        if (value < 0) value += MOD;
        return *this;
    }
    
    mod_int& operator*=(const mod_int &other) {
        value = int((long long)value * other.value % MOD);
        return *this;
    }
    
    mod_int pow(long long p) const {
        mod_int ans(1), a(*this);
        while (p) {
            if (p & 1) ans *= a;
            a *= a;
            p /= 2;
        }
        return ans;
    }
    
    mod_int inv() const {
        return pow(MOD - 2);
    }
    
    mod_int& operator/=(const mod_int &other) {
        return *this *= other.inv();
    }
    
    friend mod_int operator+(mod_int a, const mod_int &b) { a += b; return a; }
    friend mod_int operator-(mod_int a, const mod_int &b) { a -= b; return a; }
    friend mod_int operator*(mod_int a, const mod_int &b) { a *= b; return a; }
    friend mod_int operator/(mod_int a, const mod_int &b) { a /= b; return a; }
    
    bool operator==(const mod_int &other) const { return value == other.value; }
    bool operator!=(const mod_int &other) const { return value != other.value; }
    bool operator<(const mod_int &other) const { return value < other.value; }
    bool operator>(const mod_int &other) const { return value > other.value; }
    bool operator<=(const mod_int &other) const { return value <= other.value; }
    bool operator>=(const mod_int &other) const { return value >= other.value; }
    
    mod_int operator&(const mod_int &other) const {
        return mod_int((long long)value & other.value);
    }
    mod_int& operator&=(const mod_int &other) {
        value &= other.value;
        return *this;
    }
    
    mod_int operator|(const mod_int &other) const {
        return mod_int((long long)value | other.value);
    }
    mod_int& operator|=(const mod_int &other) {
        value |= other.value;
        return *this;
    }
    
    mod_int operator^(const mod_int &other) const {
        return mod_int((long long)value ^ other.value);
    }
    mod_int& operator^=(const mod_int &other) {
        value ^= other.value;
        return *this;
    }
    
    mod_int operator<<(int shift) const {
        return mod_int(((long long)value << shift) % MOD);
    }
    mod_int& operator<<=(int shift) {
        value = int(((long long)value << shift) % MOD);
        return *this;
    }
    
    mod_int operator>>(int shift) const {
        return mod_int(value >> shift);
    }
    mod_int& operator>>=(int shift) {
        value >>= shift;
        return *this;
    }
    
    friend std::ostream& operator<<(std::ostream &os, const mod_int &a) {
        os << a.value;
        return os;
    }
    
    friend std::istream& operator>>(std::istream &is, mod_int &a) {
        long long v;
        is >> v;
        a = mod_int(v);
        return is;
    }
};


using mint = mod_int<998244353>;
using vmint = vt<mint>;
using vvmint = vt<vmint>;
using vvvmint = vt<vvmint>;

typedef complex<double> cd;
template<class T>
struct Convolution { // given 2 array a and b, compute the number of pair that make up sum == x for every x from 1 to MX
    static void fft(vt<cd>& a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            double angle = 2 * M_PI / len * (invert ? -1 : 1);
            cd wlen(cos(angle), sin(angle));
            for (int i = 0; i < n; i += len) {
                cd w(1);
                for (int j = 0; j < len / 2; j++) {
                    cd u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) {
            for (cd & x : a) x /= n;
        }
    }
 
    static vt<T> convolve(const vt<T>& a, const vt<T>& b) {
        vt<cd> fa(all(a)), fb(all(b));
        int n = 1;
        while (n < (int)a.size() + (int)b.size() - 1) n <<= 1;
        fa.rsz(n);
        fb.rsz(n);
        fft(fa, false);
        fft(fb, false);
        for (int i = 0; i < n; i++) fa[i] *= fb[i];
        fft(fa, true);
        vt<T> res(n);
        for (int i = 0; i < n; i++)
            res[i] = (T)round(fa[i].real());
        return res;
    }
};

struct BigCalculator {
    std::string num1;
    std::string num2;

    BigCalculator(const std::string &n1, const std::string &n2)
        : num1(n1), num2(n2) {}

    std::string add() {
        return bigAdd(num1, num2);
    }

    std::string minus() {
        return bigSub(num1, num2);
    }

    std::string multiply() {
        return bigMul(num1, num2);
    }

private:
    static std::string bigAdd(const std::string &a, const std::string &b) {
        int i = a.size() - 1;
        int j = b.size() - 1;
        int carry = 0;
        std::string result;
        while (i >= 0 || j >= 0 || carry) {
            int sum = carry;
            if (i >= 0) {
                sum += a[i] - '0';
                i--;
            }
            if (j >= 0) {
                sum += b[j] - '0';
                j--;
            }
            carry = sum / 10;
            result.push_back('0' + sum % 10);
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    static int compare(const std::string &a, const std::string &b) {
        if (a.size() != b.size())
            return (a.size() < b.size() ? -1 : 1);
        return a.compare(b);
    }

    static std::string bigSub(const std::string &a, const std::string &b) {
        if (compare(a, b) == 0)
            return "0";
        bool negative = false;
        std::string A = a, B = b;
        if (compare(a, b) < 0) {
            negative = true;
            std::swap(A, B);
        }
        int i = A.size() - 1;
        int j = B.size() - 1;
        int carry = 0;
        std::string result;
        while (i >= 0) {
            int diff = (A[i] - '0') - carry;
            if (j >= 0) {
                diff -= (B[j] - '0');
                j--;
            }
            if (diff < 0) {
                diff += 10;
                carry = 1;
            } else {
                carry = 0;
            }
            result.push_back('0' + diff);
            i--;
        }
        while (result.size() > 1 && result.back() == '0')
            result.pop_back();
        std::reverse(result.begin(), result.end());
        if (negative)
            result.insert(result.begin(), '-');
        return result;
    }

    typedef std::complex<double> cd;

    static void fft(std::vector<cd> &a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
                std::swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2 * M_PI / len * (invert ? -1 : 1);
            cd wlen(cos(ang), sin(ang));
            for (int i = 0; i < n; i += len) {
                cd w(1);
                for (int j = 0; j < len / 2; j++) {
                    cd u = a[i + j];
                    cd v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) {
            for (int i = 0; i < n; i++)
                a[i] /= n;
        }
    }

    static std::vector<int> multiplyFFT(const std::vector<int> &a, const std::vector<int> &b) {
        std::vector<cd> fa(a.begin(), a.end());
        std::vector<cd> fb(b.begin(), b.end());
        int n = 1;
        while (n < (int)a.size() + (int)b.size())
            n <<= 1;
        fa.resize(n);
        fb.resize(n);
        fft(fa, false);
        fft(fb, false);
        for (int i = 0; i < n; i++)
            fa[i] *= fb[i];
        fft(fa, true);
        std::vector<int> result(n);
        int carry = 0;
        for (int i = 0; i < n; i++) {
            long long t = static_cast<long long>(round(fa[i].real())) + carry;
            result[i] = t % 10;
            carry = t / 10;
        }
        while (carry) {
            result.push_back(carry % 10);
            carry /= 10;
        }
        while (result.size() > 1 && result.back() == 0)
            result.pop_back();
        return result;
    }

    static std::string bigMul(const std::string &a, const std::string &b) {
        if (a == "0" || b == "0")
            return "0";
        std::vector<int> A(a.size()), B(b.size());
        for (size_t i = 0; i < a.size(); i++)
            A[a.size() - 1 - i] = a[i] - '0';
        for (size_t i = 0; i < b.size(); i++)
            B[b.size() - 1 - i] = b[i] - '0';
        std::vector<int> resultVec = multiplyFFT(A, B);
        std::string result;
        for (int i = resultVec.size() - 1; i >= 0; i--)
            result.push_back(resultVec[i] + '0');
        size_t pos = result.find_first_not_of('0');
        if (pos != std::string::npos)
            return result.substr(pos);
        else
            return "0";
    }
};