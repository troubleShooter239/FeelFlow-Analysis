using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace FeelFlowAnalysis.Models.Entities;

[BsonIgnoreExtraElements]
public class User : IUser
{
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id { get; set; } = string.Empty;
    [BsonElement("email")]
    public string Email { get; set; } = string.Empty;
    [BsonElement("password_hash")]
    public string PasswordHash { get; set; } = string.Empty;
    [BsonElement("card_details")]
    public CreditCard CardDetails { get; set; }
}