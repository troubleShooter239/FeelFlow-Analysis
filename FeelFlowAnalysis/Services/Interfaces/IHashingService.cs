namespace FeelFlowAnalysis.Services.Interfaces;

// Summary:
//     Interface for hashing operations.
public interface IHashingService
{
    // Summary:
    //     Hashes a string.
    //
    // Parameters:
    //   password:
    //     The password to hash.
    //
    // Returns:
    //     The hashed password.
    string HashString(string password);

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
    bool VerifyString(string storedPassword, string enteredPassword);
}