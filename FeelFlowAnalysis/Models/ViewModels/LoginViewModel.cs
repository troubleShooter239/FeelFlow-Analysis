using System.ComponentModel.DataAnnotations;

namespace FeelFlowAnalysis.Models.ViewModels;

public class LoginViewModel
{
    [Required(AllowEmptyStrings = false, ErrorMessage = "Please enter username")]
    public required string UserName { get; set; }

    [Required(AllowEmptyStrings = false, ErrorMessage = "Please enter password")]
    public required string Password { get; set; }
}
