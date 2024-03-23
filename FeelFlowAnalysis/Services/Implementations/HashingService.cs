using System.Security.Cryptography;
using FeelFlowAnalysis.Models.Settings;
using FeelFlowAnalysis.Services.Interfaces;

namespace FeelFlowAnalysis.Services.Implementations;

// Summary:
//     Provides hashing services.
public class HashingService(IHashingSettings settings) : IHashingService
{
    private readonly int _saltSize = settings.SaltSize;
    private readonly int _hashSize = settings.HashSize;
    private readonly int _iterations = settings.Iterations;

    // Summary:
    //     Hashes a string.
    //
    // Parameters:
    //   password:
    //     The password to hash.
    //
    // Returns:
    //     The hashed password.
    public string HashString(string password)
    {
        byte[] salt = new byte[_saltSize];
        
        using (var rng = RandomNumberGenerator.Create())
            rng.GetBytes(salt);

        byte[] hash;

        // Using PBKDF2 with SHA256 for password hashing
        using (var pbkdf2 = new Rfc2898DeriveBytes(password, salt, _iterations, HashAlgorithmName.SHA256))
            hash = pbkdf2.GetBytes(_hashSize);

        // Combine salt and hash and convert to Base64 string
        return Convert.ToBase64String(salt) + ":" + Convert.ToBase64String(hash);
    }

    // Summary:
    //     Verifies if entered password matches stored password hash.
    //
    // Parameters:
    //   storedPassword:
    //     The hashed password stored in a secure manner.
    //
    //   enteredPassword:
    //     The password entered by the user.
    //
    // Returns:
    //     true if entered password matches stored password hash; otherwise, false.
    public bool VerifyString(string storedPassword, string enteredPassword)
    {
        // Split stored password into salt and hash parts
        var passwordParts = storedPassword.Split(':');
        var salt = Convert.FromBase64String(passwordParts[0]);
        var hash = Convert.FromBase64String(passwordParts[1]);

        // Using PBKDF2 with SHA256 for password verification
        using var pbkdf2 = new Rfc2898DeriveBytes(enteredPassword, salt, _iterations, HashAlgorithmName.SHA256);
        // Compare the derived hash with the stored hash
        return Enumerable.SequenceEqual(pbkdf2.GetBytes(_hashSize), hash);
    }
}
