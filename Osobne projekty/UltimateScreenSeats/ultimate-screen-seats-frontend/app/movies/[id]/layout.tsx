"use client"

export default function MovieLayout({ children, backgroundImage }: { children: React.ReactNode, backgroundImage: string }) {
    return (
        <div className="inset-0 w-full h-full ">
            <div className="absolute inset-0 w-full h-full z-0">
                {backgroundImage ? (
                    <div
                    className="min-h-screen w-full flex flex-col bg-cover bg-center fixed"
                    style={{
                      backgroundImage: `url(${backgroundImage})`,
                      backgroundAttachment: "fixed",
                    }}
                  />
                ) : (
                    <div className="fixed inset-0 bg-gray-800" />
                )}
                <div className="fixed inset-0 bg-black bg-opacity-70" />
            </div>
            <div className="relative">{children}</div>
        </div>
    );
}