namespace FeelFlowAnalysis.Services.Interfaces;

// Summary:
//     Interface for encryption operations.
public interface IEncryption
{
    // Summary:
    //     Decrypts an encrypted string.
    //
    // Parameters:
    //   encryptedValue:
    //     The encrypted string to decrypt.
    //
    // Returns:
    //     The decrypted string.
    public string DecryptString(string encryptedValue);

    // Summary:
    //     Encrypts a string value.
    //
    // Parameters:
    //   value:
    //     The string value to encrypt.
    //
    // Returns:
    //     The encrypted string.
    string EncryptString(string value);

    // Summary:
    //     Verifies if entered data matches stored data.
    //
    // Parameters:
    //   storedData:
    //     The data stored in a secure manner.
    //
    //   enteredData:
    //     The data entered by the user.
    //
    // Returns:
    //     true if entered data matches stored data; otherwise, false.
    bool VerifyString(string storedData, string enteredData);
}
