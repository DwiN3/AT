import toast from "react-hot-toast";


export const showToast = async (message: string, isError: boolean) => {
    toast(message,
        {
            icon: isError ? '😿' : '💪',
            style: {
                borderRadius: '16px',
                textAlign: "center",
                padding: '16px',
                background: isError ? "#F31260" : "#006FEE",
                color: '#fff',
            },
        }
    );
}