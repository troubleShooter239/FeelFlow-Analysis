using FeelFlowAnalysis.Models.Enums;
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace FeelFlowAnalysis.Models.Entities;

// Summary:
//     Represents a user entity.
[BsonIgnoreExtraElements]
public class User
{
    // Summary:
    //     Gets or sets the ID of the user.
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id { get; set; } = string.Empty;
    
    // Summary:
    //     Gets or sets the email of the user.
    [BsonElement("email")]
    public string Email { get; set; } = string.Empty;

    // Summary:
    //     Gets or sets the password hash of the user.
    [BsonElement("password_hash")]
    public string PasswordHash { get; set; } = string.Empty;

    // Summary:
    //     Gets or sets the credit card details of the user.
    [BsonElement("card_details")]
    public CreditCard CardDetails { get; set; }

    // Summary:
    //     Gets or sets the role of the user.
    [BsonElement("role")]
    public UserRoles Role { get; set; }
}